import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import FastICA
from scipy.signal import medfilt
from scipy.stats import kurtosis
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from collections import deque
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import random


# ICA Class #

class ICA:
    def __init__(self, n_sensors=2):
        self.X_clean = np.zeros((3, 3*n_sensors))
        self.noise_idx = np.zeros((3*(n_sensors - 1)))
        self.C = np.zeros(3*n_sensors)
        self.U = self.X_clean


    # finds step noise and diminishes it
    # inputs: x -- Nx3 measurement matrix
    #         sigma -- gaussian std
    #         dt_max -- maximum time step
    #         tolerance -- leeway we have for finding step noise

    def correct_baseline(self, x, sigma, dt_max, tolerance, w=5):    
        
        # apply gaussian filter to simplify measurements
        y = gaussian_filter1d(x, sigma=sigma, order=2)
        
        # find zero crossings / transition points (should find where the sharp
        # vertical jumps are)
        zero_crossings = np.where(np.diff(np.sign(y)) != 0)[0]

        jump_queue = deque()
        corrected_x = x.copy()

        # find where the jumps are and, if the jumps are significant enough
        # compared to the tolerance level, offset the measurement by the jump
        # amount
        for i in zero_crossings:
            if i >= len(x) - 1:
                break

            before = np.mean(x[max(0, i - w):i]) if i > 0 else x[i]
            after = np.mean(x[i + 1:i + 1 + w])
            jump = after - before

            while jump_queue and i - jump_queue[0][0] > dt_max:
                jump_queue.popleft()

            cum_sum = sum(mag for _, mag in jump_queue)

            if jump_queue and abs(jump + cum_sum) / max(abs(cum_sum), 1e-9) <= tolerance:
                current_offset = 0.0
                while jump_queue:
                    t_idx, mag = jump_queue.popleft()
                    current_offset += mag
                    next_t_idx = jump_queue[0][0] if jump_queue else i
                    corrected_x[t_idx:next_t_idx] -= current_offset
            else:
                jump_queue.append((i, jump))

        return corrected_x


    # remove long-period trend

    # inputs:
    #    X -- Nxn measurement matrix
    #    window -- moving average window length (samples)
    #    n_passes -- number of times to apply the moving average

    # returns:
    #    trend -- Nxn trend
    #    X_detrended -- Nxn X - trend

    def detrend(self, X, window=500, n_passes=3):
    
        trend = np.copy(X)

        # window for moving average
        kernel = np.ones(window) / window
        
        for _ in range(n_passes):
            for i in range(trend.shape[1]):
                # Use padding to minimize edge distortion
                padded = np.pad(trend[:, i], window // 2, mode='edge')
                smoothed = np.convolve(padded, kernel, mode='valid')

                # Make it its original length if there's a mismatch
                trend[:, i] = smoothed[:len(X)]
                
        return trend, X - trend


    # c coefficient for each independent component, w/ C_i = sigma_i / ||mu_i||
    #  mu_i being mean of mixing vectors a_i^(j) across sensors j and 
    #  sigma_i being sqrt( (1/M) * sum_j ||a_i^(j) - mu_i||^2 )

    # if c is small, then the IC is spatially uniform (true signal)
    # if c is large, then the IC is spatially variable (stray signal)

    # inputs:
    #    A  -- nxn mixing matrix from FastICA
    #    n_sensors -- # of magnetometers
    #    n_axes -- axes per sensor

    # returns
    #    c -- (n,) array of c coefficients

    def c_coefficient(self, A, n_sensors, n_axes=3):
        n_ic = A.shape[1]
        C = np.zeros(n_ic)

        for i in range(n_ic):
            col  = A[:, i]
            summing = 0

            # find the mean of each of the sensor measurements
            vecs = [col[j * n_axes :(j+1) * n_axes] for j in range(n_sensors)]
            mu   = np.mean(vecs, axis=0)
            
            # calculate sigma_i
            for j in range(n_sensors):
                vec_j = vecs[j]
                normed = np.linalg.norm(vec_j - mu) ** 2
                summing += normed
            
            mu_norm = np.linalg.norm(mu)
            if mu_norm < 1e-12:
                C[i] = np.inf
                continue
                
            sigma = np.sqrt(summing / n_sensors)
            C[i] = sigma / mu_norm
            
        return C

    # split ICs into signals and noises by finding largest relative jump
    # in C values and split at that value -- need at least the number of
    # axes 
    
    # inputs:
    #    C  -- (n,) array of c coefficients
    #    n_axes -- # axes

    # returns:
    #    signal_idx -- indices of ICs classified as signal
    #    noise_idx  -- indices of ICs classified as noise

    def split_signal_noise(self, C, n_axes=3):

        n_ic = len(C)
        n_axes = min(n_axes, n_ic - 1)

        sorted_idx = np.argsort(C)
        sorted_C   = C[sorted_idx]

        eps = 1e-12
        best_split = n_axes
        best_ratio = -np.inf

        for k in range(n_axes, n_ic):
            ratio = (sorted_C[k] + eps) / (sorted_C[k - 1] + eps)
            if ratio > best_ratio:
                best_ratio = ratio
                best_split = k

        signal_idx = sorted_idx[:best_split]
        noise_idx  = sorted_idx[best_split:]

        return signal_idx, noise_idx
    

    # run ICA

    # inputs:
    #    X  -- (N, n_sensors * n_axes) signal matrix
    #    n_sensors -- # of magnetometers
    #    n_axes -- # axes per sensor
    #    trend_window, trend_passes -- detrending parameters
    #    n_components -- number of ICA components to estimate.
    #                     Defaults to n_sensors*n_axes (one per
    #                     observed channel). Use
    #                     inspect_eigenspectrum() to help pick a
    #                     smaller value when the data's true rank is
    #                     lower
    #    random_state -- FastICA seed

    # returns:
    #    X_clean -- (N, n_sensors * n_axes) cleaned data
    #    F -- (n, n) filter matrix
    #    C -- c coefficients
    #    signal_idx -- signal IC indices
    #    noise_idx  -- noise IC indices
    #    U -- (N, n_components) raw ICA components
    #    trend -- (N, n) trend
    #
    # also sets:
    #    self.A_mix -- (n, n_components) ICA mixing matrix
    #    self.signal_idx, self.noise_idx
    #    self.R_rel -- dict of relative calibration matrices from
    #                  relative_calibration() (sensor 0 as reference)

    def ica(self, X, n_sensors, n_axes=3, trend_window=500, trend_passes=3,
            n_components=None, kurtosis_threshold=10.0, random_state=0):

        n_obs = n_sensors * n_axes
        if n_components is None:
           n_components = n_obs

        n_sig = n_axes
        n_noise = n_obs - n_sig
        
        result = np.zeros((len(X[:, 0]), n_obs))

        for i in range(n_obs):
            filt_sig = self.correct_baseline(X[:, i], 15, 100, 0.3)
            filt_sig = signal.medfilt(filt_sig, kernel_size=51)
            if i == 0:
                result = filt_sig
            else:
                result = np.vstack((result, filt_sig))

        X_flat = result.T

        # detrend data
        trend, X_det = self.detrend(X_flat, window=trend_window, n_passes=trend_passes)

        # implement FastICA
        ica = FastICA(n_components=n_components, random_state=random_state,
                      max_iter=10000, tol=1e-3, fun='logcosh')

        self.U = ica.fit_transform(X_det)
        W = ica.components_     # separating matrix
        A = ica.mixing_         # mixing matrix
        self.A_mix = A

        # find c's
        self.C = self.c_coefficient(A, n_sensors=n_sensors, n_axes=n_axes)

        # split into signal vs noise ICs
        #signal_idx_c, noise_idx_c = self.split_signal_noise(self.C, n_axes=n_axes)

        sorted_idx  = np.argsort(self.C)
        signal_idx  = sorted_idx[:n_sig]    # find smallest 3 c's
        self.noise_idx   = sorted_idx[n_sig:]    # find rest of c's (largest)

        self.signal_idx = signal_idx

        U_clean = self.U.copy()
        U_clean[:, self.noise_idx] = 0.0

        # cleaned matrix w/ W (linear part only -- de-spiking isn't 
        # in F. self.X_clean includes both; F only has zeroing by ICs)
        A_clean = A.copy()
        A_clean[:, self.noise_idx] = 0.0
        F = A_clean @ W

        # restore trend, using the (zeroed + de-spiked) ICs directly
        self.X_clean = U_clean @ A.T + ica.mean_ + trend

        return self.X_clean, F, self.C, signal_idx, self.noise_idx, self.U, trend

    def print_info(self, X_obs, true_B, n_sensors=2):
        print('C coefficients:')
        for i in range(n_sensors*3):
            tag = 'NOISE' if i in self.noise_idx else 'signal'
            print(f'  IC{i+1}: C={self.C[i]:.4f}  [{tag}]')

        rmse_raw   = np.sqrt(np.mean((X_obs[:, :3] - true_B)**2))
        rmse_clean = np.sqrt(np.mean((self.X_clean[:, :3] - true_B)**2))
        print(f'\nRMSE raw:     {rmse_raw:.1f} nT')
        print(f'RMSE cleaned: {rmse_clean:.1f} nT')
        print(f'Improvement:  {(1 - rmse_clean/rmse_raw)*100:.1f}%')

    # the jump in the signal between a baseline window and 
    # an event window. A step-like stray field shows up as a level shift here; 
    # if it has been removed, the jump should shrink toward whatever natural 
    # variation occurred over that interval (not toward zero)

    # inputs:
    #    signal -- (N,) 1D array
    #    event_window -- (start, end) indices spanning the
    #                     suspected step
    #    baseline_window -- (start, end) indices of a nearby
    #                        quiet/reference period

    # returns:
    #    jump -- mean(signal[event_window]) - mean(signal[baseline_window])

    def jump_size(self, signal, event_window, baseline_window):

        a, b = event_window
        c, d = baseline_window
        return signal[a:b].mean() - signal[c:d].mean()


    # evaluate noise reduction via inter-sensor differences and step
    # jump sizes

    # inputs:
    #    X_obs -- (N, n_sensors*n_axes) measurement matrix
    #    n_sensors, n_axes -- # sensors and # axes

    # returns: dict with keys
    #    'pairwise' -- dict of {(i,j): {'raw_std','clean_std','reduction_pct'}}
    #                  for each sensor pair (std of X_i - X_j per axis)
    #    'events'   -- list of dicts (only if `events` given), each
    #                   with 'name','raw_jump','clean_jump'

    def evaluate_noise_reduction(self, X_obs, n_sensors=2, n_axes=3):

        result = {'pairwise': {}, 'events': []}

        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                raw_std = np.zeros(n_axes)
                clean_std = np.zeros(n_axes)
                for ax in range(n_axes):
                    ci, cj = i * n_axes + ax, j * n_axes + ax
                    diff_raw = X_obs[:, ci] - X_obs[:, cj]
                    diff_clean = self.X_clean[:, ci] - self.X_clean[:, cj]
                    raw_std[ax] = diff_raw.std()
                    clean_std[ax] = diff_clean.std()
                reduction_pct = (1 - clean_std / raw_std) * 100
                result['pairwise'][(i, j)] = {
                    'raw_std': raw_std,
                    'clean_std': clean_std,
                    'reduction_pct': reduction_pct,
                }

        print('Noise-reduction evaluation (Imajo et al. 2021 style)')
        for (i, j), d in result['pairwise'].items():
            print(f'  sensor {i+1} - sensor {j+1} difference std per axis:')
            for ax in range(n_axes):
                print(f'    axis {ax+1}: raw={d["raw_std"][ax]:.2f}  '
                        f'clean={d["clean_std"][ax]:.2f}  '
                        f'reduction={d["reduction_pct"][ax]:.1f}%')
        for ev in result['events']:
            print(f'  event "{ev["name"]}" jump: raw={ev["raw_jump"]:.2f}  '
                    f'clean={ev["clean_jump"]:.2f}')

        return result


    def print_graph(self, X_obs, true_B=None):
        axes_lbl  = ['X', 'Y', 'Z']
        show = 600

        fig = plt.figure(figsize=(18, 20))
        gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle(
            'ICA Noise Separation',
            fontsize=20, fontweight='bold', y=0.93
        )

        # Row 1: sensor 1
        for i in range(3):
            ax = fig.add_subplot(gs[0, i])
            ax.plot(X_obs[:show, i], lw=0.7, alpha=0.7, label='= Measured S1')
            ax.plot(self.X_clean[:show, i], lw=0.9, alpha=0.9, label='Cleaned S1')
            if true_B is not None:
                ax.plot(true_B[:show, i], lw=1.2, ls='--', label='True B')
            ax.set_title(f'S1 {axes_lbl[i]}-axis', fontsize=10)
            ax.set_xlabel('Sample'); ax.set_ylabel('Field (nT)')
            ax.legend(fontsize=7)

        # Rows 2-3: all 6 ICs
        for i in range(6):
            if i < 3:
                row = 1
            else:
                if i > 2 and i < 6:
                    row = 2
                else:
                    row = 3 
            ax = fig.add_subplot(gs[row, i % 3])
            ax.plot(self.U[:show, i], lw=0.7, alpha=0.9)
            is_n = i in self.noise_idx
            ax.set_title(f'IC{i+1}  [{"NOISE" if is_n else "SIGNAL"}]', fontsize=10)
            ax.set_xlabel('Sample'); ax.set_ylabel('Amplitude')
            ax.text(0.97, 0.90, f'C={self.C[i]:.3f}', transform=ax.transAxes,
                    ha='right', fontsize=9)

        # Row 4: C coefficient bar chart
        ax = fig.add_subplot(gs[3, :])
        n_ic = len(self.C)
        colors = ['tab:blue' if i not in self.noise_idx else 'tab:red'
                  for i in range(n_ic)]
        bars = ax.bar([f'IC{i+1}' for i in range(n_ic)], self.C, color=colors, alpha=0.85)

        # threshold sits between the largest 'signal' C and the smallest 'noise' C
        signal_C = [self.C[i] for i in range(n_ic) if i not in self.noise_idx]
        noise_C  = [self.C[i] for i in self.noise_idx]
        if signal_C and noise_C:
            thresh = (max(signal_C) + min(noise_C)) / 2
        else:
            thresh = float(np.median(self.C))
        ax.axhline(thresh, lw=1.5, ls='--', alpha=0.8, color='k')
        ax.set_title('C Coefficient', fontsize=11)
        ax.set_ylabel('C = σ_i / ||μ_i||')
        for bar, cv in zip(bars, self.C):
            ax.text(bar.get_x() + bar.get_width()/2, cv + 0.003,
                        f'{cv:.3f}', ha='center', fontsize=9)
        ax.legend(handles=[
            mpatches.Patch(color='tab:blue', label='Signal IC'),
            mpatches.Patch(color='tab:red', label='Noise IC'),
            mlines.Line2D([], [], ls='--', color='k', label=f'Threshold C={thresh:.3f}'),
        ], fontsize=9)

        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        plt.show()
