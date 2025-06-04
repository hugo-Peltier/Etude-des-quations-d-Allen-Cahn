# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:51:19 2025

@author: hugop
"""

import numpy as np
import matplotlib.pyplot as plt
Lx, Ly = 10.0, 10.0
Nx, Ny = 50, 50
dx, dy = Lx / Nx, Ly / Ny
epsilon = 0.01
dt = 0.01
Tmax = 8.0
steps = int(Tmax / dt)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
u = np.random.uniform(-0.5, 0.5, (Nx, Ny))
def laplacian(u):
    return (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
    ) / dx**2
def energy(u):
    grad_u_x = (np.roll(u, -1, axis=0) - u) / dx
    grad_u_y = (np.roll(u, -1, axis=1) - u) / dy
    grad_norm_sq = grad_u_x**2 + grad_u_y**2
    F_u = 0.25 * (u**2 - 1)**2
    return np.sum(0.5 * epsilon * grad_norm_sq + F_u) * dx * dy
energies = []
times = []
for n in range(steps):
    lap_u = laplacian(u)
    u = u + dt * (epsilon * lap_u - (u**3 - u))
    if n % 10 == 0:
        energies.append(energy(u))
        times.append(n * dt)
plt.figure(figsize=(8,5))
plt.plot(times, energies, label="energie libre", color="blue")
plt.xlabel("Temps")
plt.ylabel("energie")
plt.title("evolution de l'energie libre au cours du temps (semi-implicite)")
plt.grid(True)
plt.legend()
plt.savefig("courbe_energie_semi_implicite.png")
plt.show()