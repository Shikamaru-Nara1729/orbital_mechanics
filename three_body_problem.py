### References used to write this code:
###
### 1. https://levelup.gitconnected.com/the-two-body-problem-in-python-6bbe4a0b2f88
### 2. https://towardsdatascience.com/how-to-animate-plots-in-python-2512327c8263
### 3. https://towardsdatascience.com/use-python-to-create-two-body-orbits-a68aed78099c
### 4. https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767
### 5. https://observablehq.com/@rreusser/periodic-planar-three-body-orbits
###
### Code is written solely for entertainment purpose and to gain a better understanding
### of the orbital equations given in the Howard Curtis book "Orbital Mechanics for
### Engineering Students".


### TODO: There's a collision happening at some later timestep, as there was a divide error,
###       which means the r2 and r3 vectors are equal. One collision check needs to be added
###       to prevent code exiting prematurely.


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from scipy.integrate import solve_ivp


plt.rcParams['axes.grid'] = True
G = 6.67408e-20  # Universal Gravitation Constant (Units: (km^3) / (kg*(s^2)))
time = np.arange(0, 480, 0.5) # time array


def dist_cubed(r):
    return np.power(np.linalg.norm(r), 3)


def three_body_eqn(t,y, m1, m2, m3):
    r1, r2, r3 = y[:3], y[3:6], y[6:9]
    r12, r13, r23 = dist_cubed(r2 - r1), dist_cubed(r3 - r1), dist_cubed(r3 - r2)

    c0 = y[9:18]
    c1 = (G * m2 * (r2 - r1) / r12) + (G * m3 * (r3 - r1) / r13)
    c2 = (G * m1 * (r1 - r2) / r12) + (G * m3 * (r3 - r2) / r23)
    c3 = (G * m1 * (r1 - r3) / r13) + (G * m2 * (r2 - r3) / r23)
    return np.concatenate((c0, c1, c2, c3))


def plot_per_frame(idx, position_vectors):
    ax.clear()
    # Plotting m1 (in red)
    ax.plot3D(position_vectors[:idx+1, 0], position_vectors[:idx+1, 1], position_vectors[:idx+1, 2], 'red')
    ax.scatter(position_vectors[idx, 0], position_vectors[idx, 1], position_vectors[idx, 2], c='red', marker='o')

    # Plotting m2 (in blue)
    ax.plot3D(position_vectors[:idx+1, 3], position_vectors[:idx+1, 4], position_vectors[:idx+1, 5], 'blue')
    ax.scatter(position_vectors[idx, 3], position_vectors[idx, 4], position_vectors[idx, 5], c='blue', marker='o')

    # Plotting m3 (in green)
    ax.plot3D(position_vectors[:idx+1, 6], position_vectors[:idx+1, 7], position_vectors[:idx+1, 8], 'green')
    ax.scatter(position_vectors[idx, 6], position_vectors[idx, 7], position_vectors[idx, 8], c='green', marker='o')

    # Plotting the centre of mass (in black)
    ax.plot3D(position_vectors[:idx+1, 18], position_vectors[:idx+1, 19], position_vectors[:idx+1, 20], 'black')
    ax.scatter(position_vectors[idx, 18], position_vectors[idx, 19], position_vectors[idx, 20], c='black', marker='x')

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
    r1, r2, r3, g = position_vectors[:idx+1, :3], position_vectors[:idx+1, 3:6], position_vectors[:idx+1, 6:9], position_vectors[:idx+1, 18:21]
    r2 = r2 - r1
    r3 = r3 - r1
    g = g - r1
    r1 = r1 - r1

    # Plotting m1 (in red)
    ax.plot3D(r1[:idx+1, 0], r1[:idx+1, 1], r1[:idx+1, 2], 'red')
    ax.scatter(r1[idx, 0], r1[idx, 1], r1[idx, 2], c='red', marker='o')

    # Plotting m2 (in blue)
    ax.plot3D(r2[:idx+1, 0], r2[:idx+1, 1], r2[:idx+1, 2], 'blue')
    ax.scatter(r2[idx, 0], r2[idx, 1], r2[idx, 2], c='blue', marker='o')
    
    # Plotting m3 (in green)
    ax.plot3D(r3[:idx+1, 0], r3[:idx+1, 1], r3[:idx+1, 2], 'green')
    ax.scatter(r3[idx, 0], r3[idx, 1], r3[idx, 2], c='green', marker='o')

    # Plotting G, the centre of mass (in black)
    ax.plot3D(g[:idx+1, 0], g[:idx+1, 1], g[:idx+1, 2], 'black')
    ax.scatter(g[idx, 0], g[idx, 1], g[idx, 2], c='black', marker='x')

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')

    # # Make axes limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
    ax.set_title('Orbits of CoM, m2 and m3 about m1 \nTime = ' + str(np.round(time[idx], decimals=2)) + ' sec')
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)


def plot_per_frame_about_g(idx, position_vectors):
    ax.clear()
    r1, r2, r3, g = position_vectors[:idx+1, :3], position_vectors[:idx+1, 3:6], position_vectors[:idx+1, 6:9], position_vectors[:idx+1, 18:21]
    r1 = r1 - g
    r2 = r2 - g
    r3 = r3 - g
    g = g - g

    # Plotting m1 (in red)
    ax.plot3D(r1[:idx+1, 0], r1[:idx+1, 1], r1[:idx+1, 2], 'red')
    ax.scatter(r1[idx, 0], r1[idx, 1], r1[idx, 2], c='red', marker='o')

    # Plotting m2 (in blue)
    ax.plot3D(r2[:idx+1, 0], r2[:idx+1, 1], r2[:idx+1, 2], 'blue')
    ax.scatter(r2[idx, 0], r2[idx, 1], r2[idx, 2], c='blue', marker='o')

    # Plotting m3 (in green)
    ax.plot3D(r3[:idx+1, 0], r3[:idx+1, 1], r3[:idx+1, 2], 'green')
    ax.scatter(r3[idx, 0], r3[idx, 1], r3[idx, 2], c='green', marker='o')

    # Plotting G, the centre of mass (in black)
    ax.plot3D(g[:idx+1, 0], g[:idx+1, 1], g[:idx+1, 2], 'black')
    ax.scatter(g[idx, 0], g[idx, 1], g[idx, 2], c='black', marker='x')

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')

    # # Make axes limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
    ax.set_title('Orbits of m1, m2 and m3 about CoM \nTime = ' + str(np.round(time[idx], decimals=2)) + ' sec')
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)


def generate_positions():
    position_vectors = []
    # body m1 initial conditions
    # m1, r10, v10 = 1e26, np.array([0, 0, 0]), np.array([10, 20, 30])  # Units: kg, km, km/s

    # # body m2 initial conditions
    # m2, r20, v20 = 1e26, np.array([1000, 0, 0]), np.array([0, 40, 0])   # Units: kg, km, km/s

    # # body m3 initial conditions
    # m3, r30, v30 = 10, np.array([-200, 0, 0]), np.array([2, 2, 2])   # Units: kg, km, km/s

#---------------------------------------------------------Some Initial Conditions for interesting orbits---------------------------------------------
    # Figure 8
    # m1 = m2 = m3 = 1.5e20  
    # r10 = 10*np.array([0.9700436, -0.24308753, 0])
    # r20 = 10*np.array([-0.9700436, 0.24308753, 0]) 
    # r30 = np.array([0, 0, 0])  
    # v10 = np.array([0.466203685, 0.43236573, 0]) 
    # v20 = np.array([0.466203685, 0.43236573, 0]) 
    # v30 = np.array([-2*0.466203685, -2*0.43236573, 0]) 

    # Three Ovals
    # m1 = m2 = m3 = 1.5e20
    # r10 = 10*np.array([-0.98926200436,0, 0])
    # r20 = 10*np.array([2.2096177241,0, 0]) 
    # r30 = 10*np.array([-1.2203557197, 0, 0])  
    # v10 = np.array([0,1.9169244185, 0]) 
    # v20 = np.array([0,0.1910268738, 0]) 
    # v30 = np.array([0,-2.1079512924, 0]) 

    # Flower
    m1 = m2 = m3 = 1.5e20
    r10 = 10*np.array([0.0132604844,0, 0])
    r20 = 10*np.array([1.4157286016,0, 0]) 
    r30 = 10*np.array([-1.4289890859, 0,0])  
    v10 = np.array([0,1.054151921, 0]) 
    v20 = np.array([0,-0.2101466639, 0]) 
    v30 = np.array([0,-0.8440052572, 0]) 
#-------------------------------------------------------------------------------------------------------------------------------------------------------
    # Input format for the initial conditions for the odeint function
    # [X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, VX1, VY1, VZ1, VX2, VY2, VZ2, VX3, VY3, VZ3]
    # [(0) (1) (2) (3) (4) (5) (6) (7) (8) (9)  (10) (11) (12) (13) (14) (15) (16) (17)]
    y0 = np.concatenate((r10, r20, r30, v10, v20, v30))
    t_span = [0, 480]  # Start and end times
    t_eval = np.linspace(t_span[0], t_span[1], len(time))  # Time points at which to store the solution

    # Solve the ODE
    sol = solve_ivp(three_body_eqn, t_span, y0, method='DOP853', t_eval=t_eval, args=(m1, m2, m3))

    if not sol.success:
        raise RuntimeError("ODE solver did not converge")
    for y_k in sol.y.T:
        r1, r2, r3 = y_k[:3], y_k[3:6], y_k[6:9]  # Positions of m1, m2 and m3
        rg = ((m1 * r1) + (m2 * r2) + (m3 * r3)) / (m1 + m2 + m3)  # Calculation for G, the centre of mass

        # position_vectors indices (yk = 0-14, rg = 18-20)
        position_vectors.append(np.concatenate((y_k, rg), axis=None))

    # return the position_vectors list as numpy array
    return np.array(position_vectors)


def visualise_trajectory(function, position_vectors):
    line_ani = animation.FuncAnimation(fig, function, interval=30, frames=len(time), fargs=(position_vectors,), repeat=False)
    plt.show()


if __name__ == "__main__":
    position_vectors = generate_positions()

    functions = [plot_per_frame, plot_per_frame_about_m1, plot_per_frame_about_g]
    for function in functions:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        visualise_trajectory(function, position_vectors)
