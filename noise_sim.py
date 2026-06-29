import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import random


# create a random 3x3 rotation matrix
def random_rotation():
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])


# Main Noise Simulator Class #

class NoiseSim:
    # initialize simulation
    # inputs: D -- 3x3 scaling matrix
    #         bias -- 3x1 bias vector
    #         true_B -- 3x1 magnetic field vector

    def __init__(self, D = np.eye(3), bias = np.zeros((3, 1)), 
                 true_B = np.zeros((3, 1))):
        self.D = D
        self.true_B = true_B
        self.measured_B = np.zeros((3, 1))
        self.bias = bias
    

    # create scaling matrix that combines regular scaling error,
    # nonorthogonality of sensors, misalignment, and soft iron
    # in the correct order
    # inputs: ksi -- 3x1 vector of scaling factors
    #         rho, lamb, phi -- angles of nonorthogonality
    #         eps -- 3x1 vector for misalignment 
    #         D_si -- 3x3 matrix for soft iron error

    def set_scaling_matrix(self, ksi, rho, lamb, phi, eps, D_si):
    
        # create scaling factor and misalignment matrix 
        D_sf = np.eye(3)
        D_m = np.eye(3)
    
        for i in range(3):
            D_sf[i][i] += ksi[i]
        
            # make skew matrix
            D_m[(i + 1)%3][(i + 2)%3] = -eps[i]
            D_m[(i + 2)%3][(i + 1)%3] = eps[i]
        

        # create nonorthogonality matrix
        D_no = [[1, 0, 0], 
                [np.sin(rho), np.cos(rho), 0],
                [np.sin(phi)*np.cos(lamb), np.sin(lamb), np.cos(phi)*np.cos(lamb)]]
    
        # make final scaling matrix
        self.D = D_sf @ D_no @ D_si @ D_m
    
        return self.D


    # create bias vector
    # inputs: bias -- 3x1 bias vector
    
    def set_bias(self, bias):
        self.bias = bias
        
        return self.bias
    
    
    # create model of magnetometer measurement w/ noise (basic)
    # inputs: A -- 3x3 attitude matrix
    #         H -- 3x1 magnetic field matrix
    #         sigma -- magnetometer Gaussian noise std
    #         D -- scaling matrix (optional)
    #         b -- 3x1 bias vector (optional)

    def measure_B_basic(self, A, H, sigma=2.0, D=None, b=None):
        if D is None:
            D = self.D
        
        if b is None:
            b = self.bias

        v = np.random.normal(0, sigma, 3)
    
        inv_scale = np.linalg.inv(D)
        inv_mag_field = inv_scale @ (A @ H.T).T
        self.true_B = A @ H.T
        
        self.measured_B = inv_mag_field + b.T + v.T
    
        return self.measured_B.T
    
    
    # magnetometer measurement model w/ MTQ and electronics noise (not just
    # plain magnetometer misalignment)
    # inputs:
    #        A -- 3x3 attitude matrix
    #        H -- 3x1 magnetic field matrix
    #        b_tv -- 3x1 time-varying bias bector
    #        sigma -- magnetometer Gaussian noise std
    #
    #        G -- 3x3 MTQ coupling matrix
    #        M -- 3x1 MTQ magnetic dipole moment vector
    #        S -- 3xC electronics coupling matrix (C = # current channels)
    #        I_elec  -- Cx1 electronics current vector

    def measure_B_full(self, A, H, b_tv, sigma=2.0, G=None, M=None, S=None, I_elec=None):
        b_total = self.bias + b_tv

        # add MTQ noise
        if G is not None and M is not None:
            b_total += G @ M

        # add electronics noise
        if S is not None and I_elec is not None:
            b_total += S @ I_elec
        
        return self.measure_B_basic(A, H, sigma, b=b_total)  
    
    
    # simulate magnetometer measurements (basic)
    # inputs: A -- 3x3 attitude matrix
    #         H -- 3x1 magnetic field matrix
    #         sigma -- scalar std dev of Gaussian noise
    #         n -- # samples
    # returns:
    #         measurements -- nx3 matrix of magnetic field measurements
    #         errors -- nx3 matrix of measurement errors
    
    def basic_mag_error(self, A, H, sigma, n=1000):

        errors = np.zeros((n, 3))
        measurements = np.zeros((n, 3))

        for i in range(n):
            meas = self.measure_B_basic(A, H, sigma)
            measurements[i] = meas
            errors[i] = meas - self.true_B

        return measurements, errors
    
    
    # visualize scaling matrix and bias vector for basic sim by simulating
    # magnetometer rotations through all orientations
    # inputs: H -- 3x1 magnetic field matrix
    #         sigma -- scalar std dev of Gaussian noise
    #         n -- # samples
    # returns:
    #         measurements -- nx3 matrix of magnetic field measurements
    #         errors -- nx3 matrix of measurement errors
    
    def plot_sphere(self, H, sigma, n=3000):
        D = self.D
        b = self.bias

        D_ideal  = self.set_scaling_matrix([0,0,0], 0, 0, 0, [0,0,0], np.eye(3))
        b_ideal = np.zeros(3)

        # find ideal, no-noise magnetometer cloud
        ideal_pts = np.zeros((n, 3))
        for i in range(n):
            A = random_rotation()
            ideal_pts[i] = self.measure_B_basic(A, H, sigma, D_ideal, b_ideal)

        fig = plt.figure(figsize=(14, 7))

        self.D = D
        self.bias = b

        measured_pts = np.zeros((n, 3))
            
        # find misaligned, biased magnetometer measurements
        for i in range(n):
            A = random_rotation()
            measured_pts[i] = self.measure_B_basic(A, H, sigma, D, b)

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.grid(True, linewidth=0.5, linestyle=':')

        # plot ideal sphere as reference
        ax.scatter(ideal_pts[:,0], ideal_pts[:,1], ideal_pts[:,2],
                s=1.5, alpha=0.25, label='Ideal/Calibrated')

        # plot misaligned measurements
        ax.scatter(measured_pts[:,0], measured_pts[:,1], measured_pts[:,2],
                    s=1.5, alpha=0.3, label="Misaligned + Biased")

        ax.legend(fontsize=7, markerscale=4, loc='upper left')
        ax.tick_params(labelsize=6)

        scale = np.max(np.abs(H)) * 1.4
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(-scale, scale)

        fig.suptitle('Effects of Magnetometer Errors on Measurements')
        plt.show()
