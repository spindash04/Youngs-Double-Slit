# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:39:31 2024

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
delta_x = x[1]-x[0] # distance between each grid point
n = len(x) # number of grid points

t = np.linspace(0, 10, 1001) # discretized time
min_t, max_t = int(t[0]), int(t[-1]) # maximum and minimum values of time

x0 = -10 # initial position of wavepacket
sigma = 2 # wavepacket width
k0 = 2 # wavepacket momentum


def gaussian(x):
    '''
    Calculates a time-independent gaussian wavepacket
    
    Args:
        x (array): grid spacing
    Returns:
        phi (array): gaussian wavepacket
    '''
    eq1 = np.exp(-(x-x0)**2/(4*(sigma)**2))
    eq2 = ((2*np.pi)**0.25)*np.sqrt(sigma)
    phi = (eq1/eq2)*np.exp(1j*k0*x)
    return phi

def analytical_prob(x, t):
    '''
    Calculates an analytical probability density for a gaussian wavepacket
    
    Args:
        x (array): grid spacing
        t (array): discretised time
    Returns:
        array: analytical probability density
    '''
    delta_t = t / (2*(sigma**2))
    eq1 = 1 / (sigma * ((2 * np.pi)**0.5) * ((1 + (delta_t**2))**0.5))
    eq2 = 2 * (sigma**2) * (1 + (delta_t**2))
    eq3 = (x - x0 - (k0 * t))**2
    return eq1 * np.exp(-eq3/eq2)
    
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

def RHS(t, phi):
    '''
    Calculates the RHS of the time dependent schrodinger equation
    
    Args:
        t (array): discretised time
        phi (array): wavepacket
    Returns:
        array: derivative of phi with respect to time
    '''
    return 1j * 0.5 * M()@phi

def normalize(phi):
    ''' 
    Normalises a given wavefunction
    
    Args:
        phi (array): wavepacket
    Returns:
        array: normalised wavepacket
    '''
    norm = np.trapz(np.abs(phi)**2, x)
    return phi / np.sqrt(norm)


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


#%% plotting static test case

# numerical values at t=0 and t=10
prob_0 = prob_density[:, 0]
prob_t = prob_density[:, -1]

# analytical values at t=0 and t=10
analytical_prob_0 = analytical_prob(x, 0)
analytical_prob_t = analytical_prob(x, 10)

# plotting numerical and analytical prob. densities
plt.plot(x, prob_0, marker='.', color='r')
plt.plot(x, prob_t, marker='.', color='b')

plt.plot(x, analytical_prob_0, color='k')
plt.plot(x, analytical_prob_t, color='k', linestyle='--')

plt.xlabel('Position [a.u.]')
plt.ylabel('Probability')
plt.legend(('t = 0 (Numeric)', 't = 10 (Numeric)', 't = 0 (Analytical)', 't = 10 (Analytical)'), loc='upper right')
plt.title('Propagation of a free gaussian wavepacket')


#%% animating test case

fig, ax = plt.subplots()

# setting initial plot for numerical probability density at t = 0
prob_plot, = ax.plot(x, prob_0, color='r') 

# plotting analytical probability density at t = 0 and t = 10
plt.plot(x, analytical_prob_0, color='k')
plt.plot(x, analytical_prob_t, color='k', linestyle='--')

def update(i):
    ''' 
    Updates y data for animation
    
    Args:
        int: index
    Returns:
        prob_plot (array): updated probability density plot
    '''
    prob_plot.set_ydata(prob_density[:, i])
    return prob_plot

ax.set_xlabel('Position [a.u.]')
ax.set_ylabel('Probability')
ax.legend(('Numerical approximation', 't = 0 (Analytical)', 't = 10 (Analytical)'), loc='upper right')
ax.set_title('Time evolution of the numerical solution')

# animating test case
prob_animation = FuncAnimation(fig, update, interval=50, frames=np.arange(0, len(t), 10), repeat=True)
#prob_animation.save('1DTestCase.gif')


#%% calculating error and SSE values over time

def SSE(diff):
    ''' 
    Calculates the sum of sqared errors (SSE)
    
    Args:
        diff (array): difference between numerical and analytical solutions
    Returns:
        int: SSE value
    '''
    return np.sum(diff**2)

fig2 = plt.figure()

# index values of desired sampling times
timestamps = [0, 100, 500, 1000]

# calculating SSE values and generating error plot for each timestamp
for i in range(len(timestamps)):
    index = timestamps[i]
    diff_val = prob_density[:, index] - analytical_prob(x, t[index])
    diff_plot = plt.plot(x, diff_val)
    SSE_val = SSE(diff_val)
    print('SSE at t='+ str(int(index/100)) +': '+ str(SSE_val))

plt.xlabel('Position [a.u.]')
plt.ylabel('Difference')
plt.legend(('t = 0', 't = 1', 't = 5', 't = 10'), loc='upper left')
plt.title('Difference between numerical and analytical solutions')
