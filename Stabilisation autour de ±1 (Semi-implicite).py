# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:53:18 2025

@author: hugop
"""

import numpy as np
import matplotlib.pyplot as plt
Lx, Ly = 10.0, 10.0
Nx, Ny = 50, 50
dx, dy = Lx / Nx, Ly / Ny
epsilon = 0.01
dt = 0.01
Tmax = 20.0
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
times = []
percent_plus1 = []
percent_minus1 = []
for n in range(steps):
    lap_u = laplacian(u)
    u = u + dt * (epsilon * lap_u - (u**3 - u))
    if n % 10 == 0:
        times.append(n * dt)
        plus1 = np.sum(u > 0.8) / (Nx * Ny) * 100
        minus1 = np.sum(u < -0.8) / (Nx * Ny) * 100
        percent_plus1.append(plus1)
        percent_minus1.append(minus1)

plt.figure(figsize=(8,5))
plt.plot(times, percent_plus1, label="Proche de +1", color='red')
plt.plot(times, percent_minus1, label="Proche de -1", color='blue')
plt.xlabel("Temps")
plt.ylabel("Pourcentage de points (%)")
plt.title("Stabilisation autour de $+1$ et $-1$ (Schema semi-implicite)")
plt.legend()
plt.grid(True)
plt.savefig("courbe_stabilisation_semi_implicite.png")
plt.show()