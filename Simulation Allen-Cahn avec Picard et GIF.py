# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:47:54 2025

@author: hugop
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
Lx, Ly = 10.0, 10.0
Nx, Ny = 50, 50
dx, dy = Lx / Nx, Ly / Ny
epsilon = 0.01
dt = 0.001
Tmax = 80.0
steps = int(Tmax / dt)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
u = np.random.uniform(-0.5, 0.5, (Nx, Ny))
def laplacian(u):
    return (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u) / dx**2
snapshots = []
capture_interval = 50
max_iter = 20
tolerance = 1e-4
for n in range(steps):
    u_old = u.copy()
    u_new = u.copy()
    for k in range(max_iter):
        lap_u = laplacian(u_new)
        f_u_old = u_old**3 - u_old
        u_next = u_old + dt * (epsilon * lap_u - f_u_old)
        if np.linalg.norm(u_next - u_new, ord=np.inf) < tolerance:
            break
        u_new = u_next
    u = u_new
    if n % capture_interval == 0:
        snapshots.append(u.copy())
fig, ax = plt.subplots(figsize=(6,5))
img = ax.imshow(snapshots[0], extent=[0, Lx, 0, Ly], origin='lower', cmap='seismic', vmin=-1, vmax=1)
def animate(i):
    img.set_data(snapshots[i])
    ax.set_title(f"Temps = {i * capture_interval * dt:.2f}")
    return [img]
ani = animation.FuncAnimation(fig, animate, frames=len(snapshots), interval=100, blit=True)
ani.save(r'C:\Users\hugop\OneDrive\Bureau\evolution_allen_cahn_picard.gif', writer='pillow', fps=10)
plt.close()
