"""
reactor0d.py
0D reactor (homogeneous, adiabatic, constant pressure) - methane ignition.

Single-step global kinetics model (Arrhenius):
    CH4 + 2 O2 -> CO2 + 2 H2O
    omega = A * exp(-Ea/(R T)) * [CH4]^a * [O2]^b

System of ODEs:
    dY_CH4/dt = -(M_CH4 / rho) * omega
    dT/dt     = (-dH_r / (rho*cp)) * omega
integrated with the explicit 4th-order Runge-Kutta method (RK4).

Kinetic parameters after Westbrook & Dryer (1981), 1-step CH4.
"""

import numpy as np
from thermo import R_UNIV, M, cp_mass, h_molar, stoich_moles

# Kinetic parameters (Westbrook-Dryer 1-step, CGS->SI units adjusted)
A_PREEXP = 1.3e8        # pre-exponential factor [ (mol/m3)^(1-a-b) / s ] (scaled)
EA = 2.026e5            # activation energy [J/mol]
EXP_FUEL = -0.3         # order with respect to fuel
EXP_OX = 1.3            # order with respect to oxygen

# Heat of reaction (lower heating value of CH4) [J/mol CH4]
def heat_of_reaction(T=298.15):
    """Enthalpy of complete combustion of 1 mole CH4 [J/mol] (negative=exo)."""
    react, prod = stoich_moles(1.0)
    Hr = sum(n*h_molar(sp, T) for sp, n in prod.items()) \
         - sum(n*h_molar(sp, T) for sp, n in react.items())
    return Hr  # ~ -8.0e5 J/mol


def make_rhs(rho, cp_eff, P, T_ref=298.15):
    """
    Build the ODE right-hand-side function for the state y=[Y_CH4, Y_O2, T].
    rho    - mixture density [kg/m3] (assumed constant)
    cp_eff - effective mass-based mixture cp [J/(kg*K)]
    """
    dHr = heat_of_reaction(T_ref)  # J/mol CH4 (<0)

    def conc(Y, sp, T):
        # molar concentration [mol/m3]
        return rho * Y / M[sp]

    def rhs(t, y):
        Y_CH4, Y_O2, T = y
        Y_CH4 = max(Y_CH4, 0.0)
        Y_O2 = max(Y_O2, 0.0)
        c_f = conc(Y_CH4, "CH4", T)
        c_o = conc(Y_O2, "O2", T)
        k = A_PREEXP * np.exp(-EA / (R_UNIV * T))
        # fuel consumption rate [mol/(m3*s)]
        omega = k * (c_f ** abs(EXP_FUEL) if EXP_FUEL >= 0
                     else (c_f + 1e-12) ** EXP_FUEL) * (c_o ** EXP_OX)
        if c_f <= 0.0 or c_o <= 0.0:
            omega = 0.0
        dY_CH4 = -M["CH4"] * omega / rho
        dY_O2 = -2.0 * M["O2"] * omega / rho
        dT = -(dHr) * omega / (rho * cp_eff)   # -dHr>0 -> heating
        return np.array([dY_CH4, dY_O2, dT])

    return rhs


def rk4_integrate(rhs, y0, t0, t_end, dt):
    """
    Explicit 4th-order Runge-Kutta method with a fixed step.
    Returns (t_array, Y_array) where Y_array has shape (n_steps, len(y0)).
    """
    n = int(np.ceil((t_end - t0) / dt))
    t = np.linspace(t0, t0 + n*dt, n + 1)
    Y = np.zeros((n + 1, len(y0)))
    Y[0] = y0
    for i in range(n):
        ti, yi = t[i], Y[i]
        k1 = rhs(ti, yi)
        k2 = rhs(ti + dt/2, yi + dt/2 * k1)
        k3 = rhs(ti + dt/2, yi + dt/2 * k2)
        k4 = rhs(ti + dt, yi + dt * k3)
        Y[i+1] = yi + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, Y


def ignition_delay(t, T, dT_thresh=400.0, T0=None):
    """Ignition delay = time at which the maximum gradient dT/dt occurs."""
    if T0 is None:
        T0 = T[0]
    dTdt = np.gradient(T, t)
    return t[np.argmax(dTdt)]


def run_ignition(phi=1.0, T_init=1200.0, P=101325.0, t_end=0.05, dt=1e-6):
    """
    Simulate ignition of a mixture with given phi and initial temperature.
    Returns a dict of results.
    """
    react, _ = stoich_moles(phi)
    n_tot = sum(react.values())
    # mole fractions -> mass fractions
    M_mix = sum(react[sp]/n_tot * M[sp] for sp in react)
    Y0 = {sp: (react[sp]/n_tot)*M[sp]/M_mix for sp in react}
    # density from the ideal gas equation of state
    rho = P * M_mix / (R_UNIV * T_init)
    # effective cp (approximation: mass-weighted average at T_init)
    cp_eff = sum(Y0[sp]*cp_mass(sp, T_init) for sp in Y0)

    rhs = make_rhs(rho, cp_eff, P)
    y0 = np.array([Y0["CH4"], Y0["O2"], T_init])
    t, Y = rk4_integrate(rhs, y0, 0.0, t_end, dt)
    tau = ignition_delay(t, Y[:, 2])
    return {"t": t, "Y_CH4": Y[:, 0], "Y_O2": Y[:, 1], "T": Y[:, 2],
            "tau_ign": tau, "rho": rho, "cp": cp_eff}


if __name__ == "__main__":
    print("dHr =", heat_of_reaction(), "J/mol")
    res = run_ignition(T_init=1300.0)
    print(f"T_init=1300K -> T_final={res['T'][-1]:.0f} K, "
          f"tau_ign={res['tau_ign']*1e3:.3f} ms")
