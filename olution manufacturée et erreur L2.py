# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:49:56 2025

@author: hugop
"""

import numpy as np
import matplotlib.pyplot as plt
epsilon = 0.01
Lx = Ly = 1.0
Nx = Ny = 50
dx = Lx / Nx
dt = 0.001
T = 1.0
Nt = int(T / dt)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
def u_exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)
def f_source(x, y, t, u_val):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y) * (1 + 2 * np.pi**2 * epsilon) + u_val**3 - u_val
u = u_exact(X, Y, 0)
for n in range(Nt):
    t = n * dt
    lap_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
             np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u) / dx**2
    src = f_source(X, Y, t, u)
    u += dt * (epsilon * lap_u - u**3 + u + src)
u_ref = u_exact(X, Y, T)
error = np.sqrt(np.sum((u - u_ref)**2) * dx * dx)
print(f"Erreur L2 entre u_num et u_exact : {error:.5e}")