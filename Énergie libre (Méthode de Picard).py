# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:51:58 2025

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
max_iter = 20
tolerance = 1e-4
energies = []
times = []
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
    if n % 10 == 0:
        energies.append(energy(u))
        times.append(n * dt)
plt.figure(figsize=(8,5))
plt.plot(times, energies, label="energie libre (Picard)", color="green")
plt.xlabel("Temps")
plt.ylabel("energie")
plt.title("evolution de l'energie libre au cours du temps (Methode de Picard)")
plt.grid(True)
plt.legend()
plt.savefig("courbe_energie_picard.png")
plt.show()
