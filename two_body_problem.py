### References used to write this code:
###
### 1. https://levelup.gitconnected.com/the-two-body-problem-in-python-6bbe4a0b2f88
### 2. https://towardsdatascience.com/how-to-animate-plots-in-python-2512327c8263
### 3. https://towardsdatascience.com/use-python-to-create-two-body-orbits-a68aed78099c
###
### Code is written solely for entertainment purpose and to gain a better understanding
### of the orbital equations given in the Howard Curtis book "Orbital Mechanics for
### Engineering Students".


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from scipy.integrate import odeint


plt.rcParams['axes.grid'] = True
G = 6.67408e-20  # Universal Gravitation Constant (Units: (km^3) / (kg*(s^2)))
time = np.arange(0, 480, 0.5) # time array


def two_body_eqm(y, t, m1, m2):
    r_norm_cubed = np.power(np.linalg.norm(y[3:6] - y[:3]), 3)
    c0 = y[6:12]
    c1 = G * m2 * ((y[3:6] - y[:3]) / r_norm_cubed)
    c2 = G * m1 * ((y[:3] - y[3:6]) / r_norm_cubed)
    return np.concatenate((c0, c1, c2))


def plot_per_frame(idx, position_vectors):
    ax.clear()
    # Plotting m1 (in red) and m2 (in blue)
    ax.plot3D(position_vectors[:idx+1, 0], position_vectors[:idx+1, 1], position_vectors[:idx+1, 2], 'red')
    ax.plot3D(position_vectors[:idx+1, 3], position_vectors[:idx+1, 4], position_vectors[:idx+1, 5], 'blue')
    ax.scatter(position_vectors[idx, 0], position_vectors[idx, 1], position_vectors[idx, 2], c='red', marker='o')
    ax.scatter(position_vectors[idx, 3], position_vectors[idx, 4], position_vectors[idx, 5], c='blue', marker='o')

    # Plotting the centre of mass (in black)
    ax.plot3D(position_vectors[:idx+1, 12], position_vectors[:idx+1, 13], position_vectors[:idx+1, 14], 'black')
    ax.scatter(position_vectors[idx, 12], position_vectors[idx, 13], position_vectors[idx, 14], c='black', marker='o')

    plt.title('Two-Body Orbit')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')

    # # Make axes limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
    ax.set_title('Trajectory \nTime = ' + str(np.round(time[idx], decimals=2)) + ' sec')
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)


def plot_per_frame_about_m1(idx, position_vectors):
    ax.clear()
    r1, r2, g = position_vectors[:idx+1, :3], position_vectors[:idx+1, 3:6], position_vectors[:idx+1, 12:15]
    r2 = r2 - r1
    g = g - r1
    r1 = r1 - r1

    # Plotting m1 (in red)
    ax.plot3D(r1[:idx+1, 0], r1[:idx+1, 1], r1[:idx+1, 2], 'red')
    ax.scatter(r1[idx, 0], r1[idx, 1], r1[idx, 2], c='red', marker='o')

    # Plotting m2 (in blue)
    ax.plot3D(r2[:idx+1, 0], r2[:idx+1, 1], r2[:idx+1, 2], 'blue')
    ax.scatter(r2[idx, 0], r2[idx, 1], r2[idx, 2], c='blue', marker='o')

    # Plotting G, the centre of mass (in black)
    ax.plot3D(g[:idx+1, 0], g[:idx+1, 1], g[:idx+1, 2], 'black')
    ax.scatter(g[idx, 0], g[idx, 1], g[idx, 2], c='black', marker='o')

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')

    # # Make axes limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
    ax.set_title('Orbits of CoM and m2 about m1 \nTime = ' + str(np.round(time[idx], decimals=2)) + ' sec')
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)


def generate_positions():
    position_vectors = []
    # body m1 initial conditions
    m1, r10, v10 = 1e26, np.array([0, 0, 0]), np.array([10, 20, 30])  # Units: kg, km, km/s

    # body m2 initial conditions
    m2, r20, v20 = 1e26, np.array([3000, 0, 0]), np.array([0, 40, 0])   # Units: kg, km, km/s

    # Input format for the initial conditions for the odeint function
    # [X1, Y1, Z1, X2, Y2, Z2, VX1, VY1, VZ1, VX2, VY2, VZ2]
    # [(0) (1) (2) (3) (4) (5)  (6)  (7)  (8)  (9) (10) (11)]
    y0 = np.concatenate((r10, r20, v10, v20))
    y = odeint(two_body_eqm, y0, time, args=(m1, m2))
    for y_k in y:
        r1, r2 = y_k[:3], y_k[3:6]  # Positions of m1 and m2
        rg = ((m1 * r1) + (m2 * r2)) / (m1 + m2)  # Calculation for G, the centre of mass

        # position_vectors indices (yk = 0-11, rg = 12-14)
        position_vectors.append(np.concatenate((y_k, rg), axis=None))

    # return the position_vectors list as numpy array
    return np.array(position_vectors)


if __name__ == "__main__":
    position_vectors = generate_positions()

    ## Plotting the Animation
    ## Plotting m1, m2 and CoM in Inertial Frame
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    line_ani = animation.FuncAnimation(fig, plot_per_frame, interval=30, frames=len(time), fargs=(position_vectors,), repeat=False)
    plt.show()

    ## Plotting CoM and m2 about m1
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    line_ani = animation.FuncAnimation(fig, plot_per_frame_about_m1, interval=30, frames=len(time), fargs=(position_vectors,), repeat=False)
    plt.show()
