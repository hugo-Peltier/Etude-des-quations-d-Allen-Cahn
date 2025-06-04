# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:49:01 2025

@author: hugop
"""
plt.figure()
plt.imshow(u, extent=[0, Lx, 0, Ly], origin='lower', cmap='seismic', vmin=-1, vmax=1)
plt.title("etat initial $t=0$")
plt.colorbar()
plt.savefig("etat_initial_picard.png")
plt.close()
snapshot_intermediate = steps // 2
snapshot_final = steps - 1
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
    if n == snapshot_intermediate:
        plt.figure()
        plt.imshow(u, extent=[0, Lx, 0, Ly], origin='lower', cmap='seismic', vmin=-1, vmax=1)
        plt.title(f"etat intermediaire $t={n*dt:.2f}$")
        plt.colorbar()
        plt.savefig("etat_intermediaire_picard.png")
        plt.close()
    if n == snapshot_final:
        plt.figure()
        plt.imshow(u, extent=[0, Lx, 0, Ly], origin='lower', cmap='seismic', vmin=-1, vmax=1)
        plt.title(f"etat final $t={n*dt:.2f}$")
        plt.colorbar()
        plt.savefig("etat_final_picard.png")
        plt.close()
