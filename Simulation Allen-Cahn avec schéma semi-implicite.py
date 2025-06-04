# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:50:41 2025

@author: hugop
"""

import numpy as np
import matplotlib.pyplot as plt
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
    return (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
    ) / dx**2
plt.figure(figsize=(6,5))
plt.imshow(u, extent=[0, Lx, 0, Ly], origin='lower', cmap='seismic', vmin=-1, vmax=1)
plt.title("etat initial $t=0$")
plt.colorbar()
plt.savefig("etat_initial.png")
plt.close()
snapshot_intermediate = steps // 2
snapshot_final = steps - 1
for n in range(steps):
    lap_u = laplacian(u)
    u = u + dt * epsilon * lap_u - dt * (u**3 - u)
    if n == snapshot_intermediate:
        plt.figure(figsize=(6,5))
        plt.imshow(u, extent=[0, Lx, 0, Ly], origin='lower', cmap='seismic', vmin=-1, vmax=1)
        plt.title(f"etat intermediaire $t={n*dt:.2f}$")
        plt.colorbar()
        plt.savefig("etat_intermediaire.png")
        plt.close()
    if n == snapshot_final:
        plt.figure(figsize=(6,5))
        plt.imshow(u, extent=[0, Lx, 0, Ly], origin='lower', cmap='seismic', vmin=-1, vmax=1)
        plt.title(f"etat final $t={n*dt:.2f}$")
        plt.colorbar()
        plt.savefig("etat_final.png")
        plt.close()