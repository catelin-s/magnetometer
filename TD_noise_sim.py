# TD_noise_sim.py
# by: Catelin Smith
# description: expanded version of NoiseSim class that incorporates
#              time-dependence, stray/step noise, and magnetic dipoles


import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

from noise_sim import NoiseSim


# Time-Dependent Noise Simulator Class #

class TDNoiseSim(NoiseSim):
    # initialize simulation
    # inputs: pos_mtq -- 3x1 mtq position vector (relative to magnetometer)
    #         m -- 3x1 dipole moment vector

    def __init__(self, sim=NoiseSim(), pos_mtq=np.array([0., 0., 0.02]), 
                 m=np.array([0., 0., 0.2])):
        self.D = sim.D
        self.bias = sim.bias
        self.pos_mtq = pos_mtq
        self.m = m
        
    
    # calculate magnetic dipole
    # inputs: r -- 3x1 position vector
    #         m -- 3x1 dipole moment vector
    # return: 3x1 magnetic dipole vector

    def dipole_field(self, r, m):
        mu0_4pi = 1e-7
        rn = np.linalg.norm(r)
    
        if rn < 1e-10: 
            return np.zeros(3)
    
        # normalize the position
        r /= rn
    
        return mu0_4pi / rn**3 * (3*np.dot(m, r)*r - m) * 1e9
    

    # simulate noise over time
    # inputs: A -- 3x3xn attitude matrix
    #         H -- nx3 planet mag field matrix
    #         G -- 3x3 MTQ coupling matrix
    #         S -- 3xC electronics coupling matrix (C = # current channels)
    #         mtq -- 3x1 mtq step signal strength vector
    #         sigma -- scalar std dev of Gaussian noise
    #         n -- # of samples
    #         step_n -- # of step noises
    #         step_start -- time step noises start at
    # returns:
    #         X_obs -- nx3 matrix of measurements
    #         true_B -- nx3 matrix of true magnetic field
     
    def simulate_orbit(self, A, H, G, S, mtq, sigma, n=5000, step_n=3, 
                       step_start=1000):
        
        # make mtq schedule via step function
        step_signal = np.zeros(n)
        for i in range(step_n):
            step_signal[step_start + 210*i:step_start + 70 + 210*i] = 1.0

        # currents vary sinusoidally w/ sunlight
        t = np.linspace(0, 2 * np.pi, n)
        I_elec = np.vstack([1.5 * np.sin(t),
                            1.2 * np.cos(t + 0.5)]).T

        X_obs = np.zeros((n, 3))
        true_B = np.zeros((n, 3))

        # take measurements at each time
        for k in range(n):
            M = step_signal[k] * mtq
            X_obs[k] = self.measure_B_full(A[:, :, k], H[k, :], 0, sigma, G=G, 
                                          M=M, S=S, I_elec=I_elec[k])
            
            true_B[k] = A[:, :, k] @ H[k, :]
 
        return X_obs, true_B
        

    # simulate noise over time
    # inputs: A -- 3x3xn attitude matrix
    #         H -- 3xn planet mag field matrix
    #         scale -- 1x7 vector containing:
    #                0: 3x1 vector ksi
    #                1 - 3: scalars rho, lambda, phi
    #                4: 3x1 vector eps
    #                5: 3x3 matrix D_si (soft iron scaling matrix)
    #         b -- 3x1 bias vector
    #         mtq -- 1x2 vector containing:
    #                0: 3x1 vector detailing magnetometer position
    #                1: 3x1 vector detailing stray sawtooth strength
    #         sigma -- scalar std dev of Gaussian noise
    #         n -- # of samples
    #         fs -- sample rate
    #         step_n -- # of step noises
    #         step_start -- time step noises start at
    #         saw_start -- step at which sawtooth noise should start
    #         saw_end -- step at which sawtooth noise should end
    # returns:
    #         X_obs -- nx3 matrix of measurements
    #         true_B -- nx3 matrix of true magnetic field

    def simulate_orbit_2(self, A, H, scale, b, mtq, sigma, n=5000, fs=50.0,
                         step_n=3, step_start=1000, saw_start=3000, saw_end = 4500):
        # establish scaling matrix and bias
        D = self.set_scaling_matrix(
                ksi=scale[0], rho=scale[1], lamb=scale[2],
                phi=scale[3], eps=scale[4], D_si=scale[5])
        
        self.bias = b
        pos_mag = mtq[0]

        # make mtq schedule and find dipole strength
        stray_mtq = self.dipole_field(pos_mag - self.pos_mtq, self.m)
        
        step_signal = np.zeros(n)
        for i in range(step_n):
            step_signal[step_start + 210*i:step_start + 70 + 210*i] = 1.0

        # create sawtooth noise
        sawtooth = np.array([2*(k/fs % 1) - 1 for k in range(n)])
        sawtooth_signal = np.zeros(n)

        for i in range(saw_start, saw_end, 100):
            sawtooth_signal[i: i + 30] = 1.0

        stray_saw = mtq[1]

        X_obs = np.zeros((n, 3))
        true_B = np.zeros((n, 3))

        # make measurements at each time
        for k in range(n):
            s = abs(stray_mtq)*step_signal[k]
            s += sawtooth_signal[k] * stray_saw*sawtooth[k]
            v = np.random.normal(0, sigma, 3)

            X_obs[k, :] = np.linalg.inv(D) @ (A[:, :, k] @ H[k, :]) + b + s + v
            true_B[k, :] = A[:, :, k] @ H[k, :]

        return X_obs, true_B
    

    # simulate noise over time
    # inputs: A -- 3x3xn attitude matrix
    #         H -- nx3 planet mag field matrix
    #         sigma -- scalar std dev of Gaussian noise
    #         n -- # of samples
    # returns:
    #         X_obs -- nx3 matrix of measurements
    #         true_B -- nx3 matrix of true magnetic field
     
    def simulate_orbit_3(self, A, H, sigma, n=5000):
        X_obs = np.zeros((n, 3))
        true_B = np.zeros((n, 3))
        
        for i in range(n):
            X_obs[i, :] = self.measure_B_basic(A[:, :, i], H[i, :], sigma)
            true_B[i, :] = self.true_B

        return X_obs, true_B
