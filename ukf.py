import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# UKF Class #


# turns quaternion to rotation matrix
# inputs: q -- quaternion vector
def q_to_R(q: np.ndarray) -> np.ndarray:

    # normalize quaternion, then calculate
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [  2*(x*y + w*z),   1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [  2*(x*z - w*y),     2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ])

# hamilton product p * q, [w, x, y, z]
def q_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


# small rotation vector to unit quaternion
def rv_to_q(rv: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rv)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / angle
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


# cholesky with back-up eigendecomp
def _safe_cholesky(M: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(M)
        vals = np.maximum(vals, 1e-12)
        return vecs @ np.diag(np.sqrt(vals))
    

# UKF: 
# jointly estimates attitude quaternion q and orbit-frame field B_orbit
# plus a slow bias correction b_tv

# inputs:
#   D_cal -- 3x3 array, optional, known calibration scaling matrix
#   b_cal -- 3x1 array, optional, known calibration bias
#  sigma_meas -- std measurement noise
#  sigma_att_walk -- std for attitude random walk
#  sigma_B_walk -- std for B-field random walk
#  sigma_b_walk -- std for bias drift
#  alpha, beta, kappa -- UKF scaling parameters.

class UKF:

    # dimensions
    N = 9

    def __init__(
        self,
        D_cal: np.ndarray = None,
        b_cal: np.ndarray = None,
        G_cal: np.ndarray = None,
        M_cal: np.ndarray = None,
        sigma_meas: float = 2.0,
        sigma_att_walk: float = 5e-3,
        sigma_B_walk: float = 0.5,
        sigma_b_walk: float = 0.05,
        alpha: float = 1e-2,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        n = self.N

        # calibration (either fill with given values or null-ish values 
        # if unknown)
        self.D_cal = np.eye(3) if D_cal is None else np.asarray(D_cal, float)
        self.b_cal = np.zeros(3) if b_cal is None else np.asarray(b_cal, float)
        self.G_cal = np.zeros((3,3)) if G_cal is None else np.asarray(G_cal, float)
        self.M_cal = np.zeros(3) if M_cal is None else np.asarray(M_cal, float)

        # reference quaternion
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

        # error state: [phi, B_orbit_est, b_tv_correction]
        self.x = np.zeros(n)

        # fill B_orbit with a guess
        self.x[3:6] = np.array([0.0, 0.0, 50.0])

        # state covariance
        self.P = np.diag(np.array([
            0.1,  0.1,  0.1,
            2500, 2500, 2500,
            25,   25,   25,
        ]))

        # process noise
        self.Q = np.diag(np.array([
            sigma_att_walk**2, sigma_att_walk**2, sigma_att_walk**2,
            sigma_B_walk**2,   sigma_B_walk**2,   sigma_B_walk**2,
            sigma_b_walk**2,   sigma_b_walk**2,   sigma_b_walk**2,
        ]))

        # measurement noise (3x3 diagonal)
        self.R = np.eye(3) * sigma_meas**2

        # UKF weights
        self.alpha = alpha
        self.beta  = beta
        self.kappa = kappa
        lam = alpha**2 * (n + kappa) - n
        self.lam = lam

        self.w_m = np.full(2*n + 1, 1.0 / (2*(n + lam)))
        self.w_c = self.w_m.copy()
        self.w_m[0] = lam / (n + lam)
        self.w_c[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    # sigma point generation using cholesky
    def _sigma_points(self) -> np.ndarray:

        n = self.N
        S = _safe_cholesky((n + self.lam) * self.P)
        pts = np.zeros((n, 2*n + 1))
        pts[:, 0] = self.x
        for i in range(n):
            pts[:, i+1]   = self.x + S[:, i]
            pts[:, i+1+n] = self.x - S[:, i]
        return pts

    # predict what the next state will be and update covariance
    def predict(self):

        pts = self._sigma_points()
        n   = self.N

        # propagated sigma points: identity map on error state
        # (phi, B_o, b_tv all persist; mean of phi stays 0 after reset)
        pts_pred = pts.copy()

        # predicted mean
        x_pred = np.einsum('i,ji->j', self.w_m, pts_pred)

        # fold the mean phi into the reference quaternion and reset
        dq = rv_to_q(x_pred[0:3])
        self.q = q_mult(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        x_pred[0:3] = 0.0

        # predicted covariance
        P_pred = self.Q.copy()
        for i in range(2*n + 1):
            d = pts_pred[:, i] - x_pred
            P_pred += self.w_c[i] * np.outer(d, d)

        self.x = x_pred
        self.P = P_pred

    # measurement model
    def _h(self, q_sig: np.ndarray, B_orbit_sig: np.ndarray,
            b_tv_sig: np.ndarray) -> np.ndarray:

        B_body = q_to_R(q_sig) @ B_orbit_sig
        return self.D_cal @ B_body + self.b_cal + b_tv_sig + self.G_cal @ self.M_cal


    # using magnetometer measurement, update and find a mag field
    # estimate w/ the current noise estimates

    # inputs: 
    #       z_meas -- 3x1 array, raw magnetometer reading
    #       d_known -- 3x1 array, optional, known disturbance (for MTQ scheduling)
    #       nis_gate -- float, optional, threshold for unknown disturbances
    def update(self, z_meas: np.ndarray, nis_gate: float = None) -> np.ndarray:

        pts = self._sigma_points()
        n   = self.N

        # propagate sigma points through the measurement model
        Z = np.zeros((3, 2*n + 1))
        for i in range(2*n + 1):
            phi_i  = pts[0:3, i]
            B_i  = pts[3:6, i]
            b_tv_i = pts[6:9, i]

            q_sig = q_mult(self.q, rv_to_q(phi_i))
            q_sig /= np.linalg.norm(q_sig)

            Z[:, i] = self._h(q_sig, B_i, b_tv_i)

        # Predicted measurement mean
        z_pred = np.einsum('i,ji->j', self.w_m, Z)

        # covariances
        R_eff = self.R
        diff = z_meas - z_pred

        # use threshold if given
        if nis_gate is not None:
            P_zz_nominal = self.R.copy()
            for i in range(2*n + 1):
                dz = Z[:, i] - z_pred
                P_zz_nominal += self.w_c[i] * np.outer(dz, dz)
            nis = diff @ np.linalg.inv(P_zz_nominal) @ diff
            if nis > nis_gate:
                # change R based on how far the difference is outside
                # of the given threshold
                R_eff = self.R * (nis / nis_gate)


        # calculate kalman gain
        P_zz = R_eff.copy()
        P_xz = np.zeros((n, 3))
        for i in range(2*n + 1):
            dz = Z[:, i] - z_pred
            dx = pts[:, i] - self.x
            P_zz += self.w_c[i] * np.outer(dz, dz)
            P_xz += self.w_c[i] * np.outer(dx, dz)

        K = P_xz @ np.linalg.inv(P_zz)

        # state update
        dx = K @ diff
        self.x += dx

        # fold attitude correction into q, reset phi
        dq_corr = rv_to_q(self.x[0:3])
        self.q = q_mult(self.q, dq_corr)
        self.q /= np.linalg.norm(self.q)
        self.x[0:3] = 0.0

        # covariance update
        self.P -= K @ P_zz @ K.T

        # symmetrise to prevent drift
        self.P = 0.5 * (self.P + self.P.T)

        # return best estimate of TRUE body-frame field
        B_body_est = q_to_R(self.q) @ self.x[3:6]
        return B_body_est

    @property
    def B_orbit_est(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def attitude_quaternion(self) -> np.ndarray:
        return self.q.copy()

    @property
    def b_tv_est(self) -> np.ndarray:
        return self.x[6:9].copy()
