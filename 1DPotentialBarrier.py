# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 00:45:26 2024

@author: 08sha
"""

## program uses units such that hbar = m = 1

#%% solving time dependent schrodinger equation 

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags, csr_matrix
from matplotlib.animation import FuncAnimation


x = np.linspace(-25, 25, 501) # grid points
delta_x = x[1]-x[0] # distance between grid points
n = len(x) # number of grid points

t = np.linspace(0, 10, 1001) # discretized time
min_t, max_t = int(t[0]), int(t[-1]) # maximum and minimum values of time

x0 = -10 # initial position of wavepacket
sigma = 2 # wavepacket width
k0 = 2 # wavepacket momentum 

barrier_height = 1 # potential barrier height 
barrier_width = 2 # potential barrier width


def gaussian(x):
    '''
    Calculates a time-independent gaussian wavepacket
    
    Args:
        x (array): grid points
    Returns:
        phi (array): 1D gaussian wavepacket
    '''
    eq1 = np.exp(-(x-x0)**2/(4*(sigma)**2))
    eq2 = ((2*np.pi)**0.25)*np.sqrt(sigma)
    phi = (eq1/eq2)*np.exp(1j*k0*x)
    return phi
    
def M():
    '''
    Generates the M matrix
    
    Returns:
        matrix: M matrix
    '''
    d = (1/(delta_x**2)) * np.ones(n)
    data = np.vstack([d, -2*d, d])
    diags = np.array([-1, 0, 1])
    m = spdiags(data, diags, n, n)
    return csr_matrix(m)

def potential():
    ''' 
    Generates a potential barrier of set height and width
    
    Returns:
        potential (array): potential within grid spacing
    '''
    global barrier_height, barrier_width
    potential = np.zeros(n)
    for i in range(0, n):
        if np.abs(x[i]) <= (barrier_width * 0.5):
            potential[i] = barrier_height
    return potential

def V():
    ''' 
    Generates the V matrix
    
    Returns:
        matrix: V matrix
    '''
    data = potential()
    diags = np.array([0])
    v = spdiags(data, diags, n, n)
    return csr_matrix(v)

def RHS(t, phi):
    '''
    Calculates the RHS of the time dependent schrodinger equation
    
    Args:
        t (array): discretised time
        phi (array): wavefunction 
    Returns:
        array: derivative of phi with respect to time
    '''
    matrix = -1j * ((-0.5 * M()) + V())
    return matrix@phi

def normalize(phi):
    ''' 
    Normalises a given wavefunction
    
    Args:
        phi (array): wavefunction  
    Returns:
        array: normalised wavefunction
    '''
    norm = np.trapz(np.abs(phi)**2, x)
    return phi / np.sqrt(norm)

def coefficients(prob):
    ''' 
    Calculates transmission and reflection coefficents for a given PD
    
    Args:
        prob (array): probability density   
    Returns:
        transmission (int): transmission coefficent
        reflection (int): reflection coefficent
    '''
    transmission = np.trapz(prob[x>=0], x[x>=0])
    reflection = (np.trapz(prob[x<0], x[x<0]))
    return transmission, reflection


# defining initial wavepacket
phi = gaussian(x) 

# solving RHS of time dependent schrodinger eq.
dpdt = solve_ivp(RHS, (min_t, max_t), y0 = phi, t_eval=t) 
wavepacket = dpdt.y 

# normalising wavepacket
for i in range(len(t)):
    wavepacket[:, i] = normalize(wavepacket[:, i])
 
# calculating probability density      
prob_density = np.abs(wavepacket)**2


#%% plotting transmission and reflection coefficents

fig = plt.figure()

# initialising arrays for T and R coefficents 
T_array = []
R_array = []

# calculating T and R coefficents over each time step
for i in range(len(t)):
    T, R = coefficients(prob_density[:, i])
    T_array.append(T)
    R_array.append(R)

# plotting T and R coefficents over time   
plt.plot(t, T_array)
plt.plot(t, R_array, color='r')

plt.ylabel('Magnitude')
plt.xlabel('t [s]')
plt.legend(('Transmission', 'Reflection'), loc='center right')
plt.title('Transmission and reflection coefficients over time')


#%% animating wavepacket and probability density 

# configuring subplots
fig2 = plt.figure()

plt.subplots_adjust(hspace=0.6)
ax1 = fig2.add_subplot(2, 1, 1)
ax2 = fig2.add_subplot(2, 1, 2)

# setting text to initial coefficent values
coeff = plt.suptitle('T = '+ str(round(T_array[0], 2)) + '   R = '+ str(round(R_array[0], 2)))

# defining scaled potential barrier
scaled_potential = potential()/4 # displayed potential can be scaled here

# plotting scaled potential barrier
potential1, = ax1.plot(x, scaled_potential, color='orange')
potential2, = ax2.plot(x, scaled_potential, color='orange')

# setting initial plots for wavepacket and probability density
phi_plot_real, = ax1.plot(x, np.real(wavepacket[:, 0]), color='r')
phi_plot_imag, = ax1.plot(x, np.imag(wavepacket[:, 0]), color='g')
prob_plot, = ax2.plot(x, prob_density[:, 0])

def update(i):
    ''' 
    Updates y data for animation
    
    Args:
        int: index
    Returns:
        phi_plot_real (array): updated real part of wavepacket
        phi_plot_imaginary (array): updated imaginary part of wavepacket
        prob_plot (array): updated probability density plot
    '''
    phi_plot_real.set_ydata(np.real(wavepacket[:, i]))
    phi_plot_imag.set_ydata(np.imag(wavepacket[:, i]))
    prob_plot.set_ydata(prob_density[:, i])
    coeff.set_text('T = '+ str(round(T_array[i], 2)) + '   R = '+ str(round(R_array[i], 2)))
    return phi_plot_real, phi_plot_imag, prob_plot

ax1.set_xlabel('Position [a.u.]')
ax1.set_ylabel('Amplitude')
ax1.legend(('Scaled potential','Real part', 'Imaginary part'), loc='upper right')
ax1.set_title('Wavefunction')

ax2.set_xlabel('Position [a.u.]')
ax2.set_ylabel('Probability')
ax2.legend(('Scaled potential','Probability density'), loc='upper right')
ax2.set_title('Probability density')

# animating wavepacket and probability density
phi_animations = FuncAnimation(fig2, update, interval=50, frames=np.arange(0, len(t), 10), repeat=True)
#phi_animations.save('1DPotentialBarrier.gif')


#%% plotting snapshot graph

fig3 = plt.figure()

# plotting scaled potential barrier
potential3, = plt.plot(x, scaled_potential, color='r')

timestamps = [0, int((len(t)-1)*0.5), int(len(t)-1)]

for i in range(len(timestamps)):
    index = timestamps[i]
    plt.plot(x, prob_density[:, index])

plt.xlabel('Position [a.u.]')
plt.ylabel('Probability')
plt.legend(('Scaled potential barrier', 't= '+ str(int(timestamps[0]/100)), 't= '+ str(int(timestamps[1]/100)), 't= '+ str(int(timestamps[2]/100))), loc='upper right')
plt.title('Probability densities at a range of times')