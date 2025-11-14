# -*- coding: utf-8 -*-

#%% solving the time dependent schrodinger equation

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags, csr_matrix
from matplotlib.animation import FuncAnimation


x = np.linspace(-25, 25, 101) # smaller grid spacing for faster computation
X, Y = np.meshgrid(x, x) # configuring 2D grid
delta_x = x[1]-x[0] # distance between grid points
n = len(x) # number of grid points in a row

t = np.linspace(0, 10, 1001) # discretised time
min_t, max_t = int(t[0]), int(t[-1]) # maximum and minimum values of time

x0 = -10 # initial position of wavepacket in x
y0 = -10 # initial position of wavepacket in y
sigma = 2 # wavepacket width
k0_x = 3 # wavepacket momentum in x direction
k0_y = 3 # wavepacket momentum in y direction

barrier_height = 10 # potential barrier height
radius = 7 # potential barrier radius


def gaussian(x, y):
    '''
    Calculates a time-independent gaussian wavepacket
    
    Args:
        x (array): grid points in x
        y (array): grid points in y
    Returns:
        phi (matrix): 2D gaussian wavepacket
    '''
    eq1 = np.exp(-((x-x0)**2 + (y-y0)**2)/(4*(sigma)**2))
    eq2 = ((2*np.pi)**0.25)*np.sqrt(sigma)
    phi = (eq1/eq2)*np.exp(1j * (k0_x*x + k0_y*y))
    return phi
     
def M():
    '''
    Generates the M matrix
    
    Returns:
        matrix: M matrix
    '''
    d = (1/(delta_x**2)) * np.ones(n*n)
    data = np.vstack([-4*d, d, d, d, d])
    diags = np.array([0, 1, -1, n, -n])
    m = spdiags(data, diags, n*n, n*n)
    return csr_matrix(m)

def potential():
    ''' 
    Generates a circular coulomb potential of set height and radius
    
    Returns:
        potential (array): potential within grid spacing
    '''
    global barrier_height, radius
    potential = np.zeros((n, n))
    r = np.sqrt((X)**2 + (Y)**2)
    potential[r <= radius] = barrier_height
    return potential.flatten()

def V():
    ''' 
    Generates the V matrix
    
    Returns:
        matrix: V matrix
    '''
    data = potential()
    diags = np.array([0])
    v = spdiags(data, diags, n*n, n*n)
    return csr_matrix(v)

def RHS(t, phi):
    '''
    Calculates the RHS of the time dependent schrodinger equation
    
    Args:
        t (array): discretised time
        phi (array): wavepacket 
    Returns:
        array: derivative of phi with respect to time
    '''
    matrix = -1j * ((-0.5 * M()) + V())
    return matrix@phi

def normalize(phi):
    ''' 
    Normalises a given wavefunction
    
    Args:
        phi (array): wavepacket  
    Returns:
        array: normalised wavepacket
    '''
    norm = np.trapz(np.abs(phi)**2, X.flatten())
    return phi / np.sqrt(norm)


# defining initial wavepacket
phi = gaussian(X.flatten(), Y.flatten()).flatten()

# solving RHS of time dependent schrodinger eq.
dpdt = solve_ivp(RHS, (min_t, max_t), y0 = phi, t_eval=t) 
wavepacket = dpdt.y.T

# normalising wavepacket
for i in range(len(t)):
    wavepacket[i, :] = normalize(wavepacket[i, :])

# calculating probability density
prob_density = np.abs(wavepacket)**2


#%% plotting animated probability density

# configuring subplots
fig = plt.figure() 
ax = fig.add_subplot(111, xlim=(x[0], x[-1]), ylim=(x[0], x[-1]))

ax.set_xlabel('X Position [a.u.]')
ax.set_ylabel('Y Position [a.u.]')
ax.set_title('2D gaussian wavepacket encountering a coulomb potential')

# plotting initial probability density map
prob_map = ax.imshow(prob_density[0, :].reshape(n, n), extent=[x[0], x[-1], x[0], x[-1]], vmin=0, vmax=np.max(prob_density), cmap='gnuplot')
plt.colorbar(prob_map, ax = ax)

def update(i):
    ''' 
    Updates density map for animation
    
    Args:
        int: index
    Returns:
        prob_map (matrix): updated probability density plot
    '''
    prob_map.set_data(prob_density[i, :].reshape(n, n))
    return prob_map

# animating probability density
prob_animation = FuncAnimation(fig, update, interval=50, frames=np.arange(0, len(t), 10), repeat=True)

#prob_animation.save('2DRutherfordScattering.gif')
