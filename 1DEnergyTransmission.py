# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:30:48 2024

@author: 08sha
"""

## program uses units such that hbar = m = 1

#%% solving time dependent schrodinger equation

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags, csr_matrix


x = np.linspace(-35, 35, 201) # grid points
delta_x = x[1]-x[0] # distance between grid points
n = len(x) # number of grid points

t = np.linspace(0, 10, 1001) # discretised time
min_t, max_t = int(t[0]), int(t[-1]) # maximum and minimum values of time

x0 = -10 # initial position of wavepacket
sigma = 2 # wavepacket width
k0 = np.linspace(1, 6, 100) # wavepacket momentum

barrier_height = 2 # potential barrier height
barrier_width = 4 # potential barrier width


def gaussian(x, k0):
    '''
    Calculates a time-independent gaussian wavepacket
    
    Args:
        x (array): grid points
        k0 (int): wavepacket momentum
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
    Calculates transmission coefficent for a given PD
    
    Args:
        prob (array): probability density   
    Returns:
        transmission (int): transmission coefficent
    '''
    transmission = np.trapz(prob[x>=0], x[x>=0])
    return transmission

def analytical(E):
    '''
    Calculates the analytical transmission coefficent 
    
    Args:
        E (int): energy of the wavefunction
    Returns:
        T (int): transmission coefficent

    '''
    if E > barrier_height:
        k1 = np.sqrt(2*(E - barrier_height)) 
        neumerator = (barrier_height**2) * (np.sin(k1*barrier_width)**2)
        denominator = 4 * E * (E - barrier_height)
        fraction = neumerator / denominator
        T = 1 / (1 + fraction)
    elif E < barrier_height:
        k1 = np.sqrt(2*(barrier_height - E))
        neumerator = (barrier_height**2) * (np.sinh(k1*barrier_width)**2)
        denominator = 4 * E * (barrier_height - E)
        fraction = neumerator / denominator
        T = 1 / (1 + fraction)
    else:
        T = 1 / (1 + (((barrier_width**2) * barrier_height)/2))
    return T
        

# initialising arrays
T_array = []
E_array = []
analytical_T_array = []

# calculating the final transmission coefficient for each wavepacket energy
for i in range(len(k0)):
    # defining initial wavepacket
    phi = gaussian(x, k0[i]) 

    # solving RHS of time dependent schrodinger eq.
    dpdt = solve_ivp(RHS, (min_t, max_t), y0 = phi, t_eval=t) 
    wavepacket = dpdt.y 

    # normalising wavepacket
    for j in range(len(t)):
        wavepacket[:, j] = normalize(wavepacket[:, j])
    
    # calculating probability density
    prob_density = np.abs(wavepacket)**2

    # calculating numerical transmission coefficient and wavepacket energy
    T = coefficients(prob_density[:, -1])
    E = (k0[i]**2) / 2
    
    # calculating analytical transmission coefficent 
    analytical_T = analytical(E)
    
    # appending values to arrays
    T_array.append(T)
    E_array.append(E)
    analytical_T_array.append(analytical_T)
 
#%% plotting transmission coefficents for a range of energies

# plotting numerical and analytical relationship
plt.plot(E_array, T_array, linestyle=':', marker='o')
plt.plot(E_array, analytical_T_array, linestyle='--')

plt.tick_params(axis='both', labelsize=16)
plt.xlabel('Initial energy [a.u.]', fontsize=20)
plt.ylabel('Transmission coeffient', fontsize=20)
plt.legend(('Numerical', 'Analytical'), loc='center right', fontsize=20)
plt.title('Transmission coefficent for a range of initial wavepacket energies')