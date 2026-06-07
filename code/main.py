"""
main.py
Main script: runs the three project parts and generates the figures,
saved to the figures/ directory (later used in the LaTeX report).

Parts:
  1. Adiabatic flame temperature T_ad(phi)  [Newton-Raphson]
  2. Ignition in a 0D reactor, T(t), Y(t), Arrhenius plot  [RK4]
  3. 1D laminar flame profile  [finite differences + Newton]
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flame_temp import adiabatic_T, sweep_phi
from reactor0d import run_ignition
from flame1d import Flame1D

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 120,
                     "lines.linewidth": 1.8, "grid.alpha": 0.3})


# ----------------------------------------------------------------------
# PART 1: T_ad vs phi
# ----------------------------------------------------------------------
def part1():
    phis = np.linspace(0.5, 1.0, 26)
    Tad = sweep_phi(phis)
    # Newton convergence for phi=1
    _, hist = adiabatic_T(1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(phis, Tad, "o-", ms=3, color="#c0392b")
    ax1.set_xlabel(r"Equivalence ratio $\phi$")
    ax1.set_ylabel(r"$T_{ad}$ [K]")
    ax1.set_title("Adiabatic flame temperature")
    ax1.grid(True)

    ks = [h[0] for h in hist]; fs = [abs(h[2]) for h in hist]
    ax2.semilogy(ks, np.maximum(fs, 1e-12), "s-", color="#2c3e50")
    ax2.set_xlabel("Newton iteration $k$")
    ax2.set_ylabel(r"$|f(T_k)|$ [J]")
    ax2.set_title("Newton method convergence ($\\phi=1$)")
    ax2.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "part1_Tad.png"))
    plt.close(fig)

    print(f"[1] T_ad(phi=1) = {Tad[-1]:.1f} K, "
          f"max @ phi={phis[np.argmax(Tad)]:.3f} -> {Tad.max():.1f} K")
    return phis, Tad


# ----------------------------------------------------------------------
# PART 2: 0D reactor
# ----------------------------------------------------------------------
def part2():
    # time history for a single temperature
    res = run_ignition(T_init=1300.0, t_end=0.05, dt=1e-6)
    t = res["t"]*1e3   # ms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(t, res["T"], color="#c0392b")
    ax1.axvline(res["tau_ign"]*1e3, ls="--", color="gray",
                label=f"$\\tau_{{ign}}$={res['tau_ign']*1e3:.2f} ms")
    ax1.set_xlabel("Time [ms]"); ax1.set_ylabel("T [K]")
    ax1.set_title("Ignition: temperature"); ax1.legend(); ax1.grid(True)

    ax2.plot(t, res["Y_CH4"], label=r"$Y_{CH_4}$", color="#2980b9")
    ax2.plot(t, res["Y_O2"], label=r"$Y_{O_2}$", color="#27ae60")
    ax2.set_xlabel("Time [ms]"); ax2.set_ylabel("Mass fraction")
    ax2.set_title("Ignition: species"); ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "part2_ignition.png"))
    plt.close(fig)

    # Arrhenius plot: ln(tau) vs 1000/T
    T_inits = np.array([1200, 1250, 1300, 1350, 1400, 1450, 1500.0])
    taus = []
    for Ti in T_inits:
        r = run_ignition(T_init=Ti, t_end=0.1, dt=2e-6)
        taus.append(r["tau_ign"])
    taus = np.array(taus)
    invT = 1000.0/T_inits
    # linear fit ln(tau)=a*(1000/T)+b -> apparent Ea
    coef = np.polyfit(invT, np.log(taus), 1)
    Ea_app = coef[0]*1000.0*8.314462618  # J/mol

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(invT, taus*1e3, "o", color="#c0392b", label="simulation")
    ax.semilogy(invT, np.exp(np.polyval(coef, invT))*1e3, "-",
                color="#2c3e50",
                label=f"fit, $E_a^{{app}}$={Ea_app/1e3:.0f} kJ/mol")
    ax.set_xlabel(r"$1000/T_{init}$ [1/K]")
    ax.set_ylabel(r"$\tau_{ign}$ [ms]")
    ax.set_title("Arrhenius plot - ignition delay")
    ax.legend(); ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "part2_arrhenius.png"))
    plt.close(fig)

    print(f"[2] tau_ign(1300K)={res['tau_ign']*1e3:.2f} ms, "
          f"Ea_app={Ea_app/1e3:.0f} kJ/mol")
    return T_inits, taus


# ----------------------------------------------------------------------
# PART 3: 1D flame
# ----------------------------------------------------------------------
def part3():
    f = Flame1D(N=201, Tu=300.0, Tb=2200.0)
    x, T, Su = f.solve(Su_guess=0.4)
    dTdx = np.gradient(T, x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(x*1e3, T, color="#c0392b")
    ax1.set_xlabel("x [mm]"); ax1.set_ylabel("T [K]")
    ax1.set_title(f"Temperature profile (delta={f.flame_thickness()*1e3:.3f} mm)")
    ax1.grid(True)

    ax2.plot(x*1e3, dTdx, color="#8e44ad")
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("dT/dx [K/m]")
    ax2.set_title("Temperature gradient")
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "part3_flame1d.png"))
    plt.close(fig)

    # grid convergence study
    Ns = [51, 101, 201, 401]
    deltas = []
    for N in Ns:
        ff = Flame1D(N=N)
        ff.solve(Su_guess=0.4)
        deltas.append(ff.flame_thickness()*1e3)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ns, deltas, "o-", color="#16a085")
    ax.set_xlabel("Number of nodes N"); ax.set_ylabel("Flame thickness [mm]")
    ax.set_title("Grid convergence")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "part3_grid.png"))
    plt.close(fig)

    print(f"[3] delta={f.flame_thickness()*1e3:.3f} mm, "
          f"Newton iters={f.iters}, grid study={['%.3f'%d for d in deltas]}")


if __name__ == "__main__":
    print("=== Numerical methods in combustion: CH4-air ===")
    part1()
    part2()
    part3()
    print("Figures saved to:", FIG)
