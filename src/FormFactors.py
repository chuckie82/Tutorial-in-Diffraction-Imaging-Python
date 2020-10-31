#
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

def evaluate_form_factor(q_sample, a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5):
    fv = (a1 * np.exp(-b1 * q_sample ** 2) +
          a2 * np.exp(-b2 * q_sample ** 2) +
          a3 * np.exp(-b3 * q_sample ** 2) +
          a4 * np.exp(-b4 * q_sample ** 2) +
          a5 * np.exp(-b5 * q_sample ** 2) + c)
    return fv

def plot_form_factors(ax, q_sample, fvC, fvN, fvO, fvP, title=None):
    ax.plot(q_sample, fvC, '.-', color='green', label='Carbon')
    ax.plot(q_sample, fvN, '.-', color='blue', label='Nitrogen')
    ax.plot(q_sample, fvO, '.-', color='red', label='Oxygen')
    ax.plot(q_sample, fvP, '.-', color='orange', label='Phosphorus')
    ax.set_ylabel('f(q)')
    ax.set_xlabel('q (inverse Angstroem)')
    if title is not None:
        ax.set_title(title)
    ax.grid()
    ax.legend()
    return ax

def get_B_attenuation_factor(q_sample, B):
    Bq = B/(8*np.pi*np.pi)
    w = np.exp(-Bq*q_sample*q_sample)
    return w

class HardSphere():
    def __init__(self, q_sample, radius=1e4, electron_density=0.344):
        """hard_sphere_form_factor:
        . q_sample (inverse Angstroem): ...
        . radius   (Angstroem): ...
        """
        self.q = q_sample
        self.R = radius
        self.r = np.linspace(0, 2*self.R, 201)
        self.rho0 = electron_density
        self.vol  = self.volume()
        self.Z    = self.number_of_electrons()
        
    def volume(self):
        return 4.*np.pi*self.R*self.R*self.R/3.
    
    def number_of_electrons(self):
        return self.rho0 * self.vol
        
    def form_factor(self):
        qR = self.q * self.R
        A_sph = 3*(np.sin(qR) - qR*np.cos(qR))/(qR**3)
        self.F = self.Z * A_sph
        return self.F
    
    def density(self):
        self.rho = self.rho0 * np.where(self.r <= self.R, 1., 0.)
        return self.rho
    
    def intensity(self):
        F = self.form_factor()
        return np.abs(F)**2

def show_sphere(q_sample, radius, intensity, size=1., color='white', title=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4), dpi=180)
    if title is not None:
        fig.suptitle(title, y = 1.03, fontsize=16)
    ax = axes[0]
    circle = plt.Circle((0,0), radius, color=color)
    ax.set_xlim((-size/2,size/2))
    ax.set_ylim((-size/2,size/2))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.add_artist(circle)
    
    ax = axes[1]
    ax.set_yscale('log')
    ax.set_xlabel('q')
    ax.set_ylabel('F^2(q)')
    ax.plot(q_sample, intensity, color=color)
    
    plt.tight_layout()

def blur_intensity(intensity, kernel=20):
    gauss = signal.hann(kernel)
    return signal.convolve(intensity, gauss, mode='same')/sum(gauss) 

def plot_hard_spheres_summary(q_sample,
                              intensity_solvated,
                              intensity_solvent,
                              intensity_solute,
                              intensity_solute_scaled,
                              R_solute=0., R_solvent=0., size=1.):
    """"""
    fig, axes = plt.subplots(nrows=2, ncols=4, sharey='row', figsize=(16,8), dpi=180)
    
    ax = axes[0,0]
    circle1 = plt.Circle((0,0), R_solvent, color='blue')
    circle2 = plt.Circle((0,0), R_solute, color='brown')
    ax.set_xlim((-size/2,size/2))
    ax.set_ylim((-size/2,size/2))
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax = axes[0,1]
    ax.set_ylabel('=', fontsize=24, rotation='horizontal', horizontalalignment='right')
    circle = plt.Circle((0,0), R_solvent, color='blue')
    ax.set_xlim((-size/2,size/2))
    ax.set_ylim((-size/2,size/2))
    ax.add_artist(circle)
    ax = axes[0,2]
    ax.set_ylabel('+', fontsize=24, rotation='horizontal', horizontalalignment='right')
    circle = plt.Circle((0,0), R_solute, color='brown')
    ax.set_xlim((-size/2,size/2))
    ax.set_ylim((-size/2,size/2))
    ax.add_artist(circle)
    ax = axes[0,3]
    ax.set_ylabel('--', fontsize=24, rotation='horizontal', horizontalalignment='right')
    circle = plt.Circle((0,0), R_solute, color='blue')
    ax.set_xlim((-size/2,size/2))
    ax.set_ylim((-size/2,size/2))
    ax.add_artist(circle)
   
    ax = axes[1,0]
    ax.set_yscale('log')
    ax.set_xlabel('q')
    ax.set_ylabel('F^2(q)')
    ax.plot(q_sample, intensity_solvated, color='black')
    ax.grid()
    ax = axes[1,1]
    ax.set_ylabel('~', fontsize=24, rotation='horizontal', horizontalalignment='right')
    ax.set_yscale('log')
    ax.set_xlabel('q')
    ax.plot(q_sample, intensity_solvent, color='blue')
    ax.grid()
    ax = axes[1,2]
    ax.set_ylabel('+', fontsize=24, rotation='horizontal', horizontalalignment='right')
    ax.set_yscale('log')
    ax.set_xlabel('q')
    ax.plot(q_sample, intensity_solute, color='brown')
    ax.grid()
    ax = axes[1,3]
    ax.set_ylabel('--', fontsize=24, rotation='horizontal', horizontalalignment='right')
    ax.set_yscale('log')
    ax.set_xlabel('q')
    ax.plot(q_sample, intensity_solute_scaled, color='blue')
    ax.grid()

    plt.tight_layout()
    plt.show()

def plot_excess_intensity(q_sample, intensity_solvated, intensity_solvent, 
                          intensity_solute, intensity_solute_scaled, intensity_cross):
    """"""
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12,6), dpi=180)
    ax = axes[0]
    ax.set_yscale('log')
    ax.plot(q_sample, intensity_solvated - intensity_solvent, label='solvated - solute-free intensity')
    ax.plot(q_sample, intensity_solute_scaled, label='rescaled solute intensity')
    ax.legend()
    ax = axes[1]
    ax.set_yscale('log')
    ax.plot(q_sample, intensity_solute, label='solute')
    ax.plot(q_sample, intensity_cross, label='cross_term')
    ax.plot(q_sample, intensity_solvent, label='solvent')
    ax.legend()
    plt.show()

def plot_babinet(q_sample, outer_shell_pattern, inner_shell_pattern,
                 R_outer=1., R_inner=0., size=1.):
    """"""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4), dpi=180)
    ax = axes[0]
    circle1 = plt.Circle((0,0), R_outer, color='blue')
    circle2 = plt.Circle((0,0), R_inner, color='brown')
    ax.set_xlim((-size/2,size/2))
    ax.set_ylim((-size/2,size/2))
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax = axes[1]
    ax.set_yscale('log')
    ax.set_ylabel('intensity')
    ax.plot(inner_shell_pattern, color='red', label='inner shell')
    ax.plot(outer_shell_pattern, color='blue', label='outer shell')
    ax.legend()
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('intensity ratio')
    ax2.plot(outer_shell_pattern/inner_shell_pattern, color='grey')
    ax2.grid()
    plt.show()
