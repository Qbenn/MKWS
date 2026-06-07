"""
flame_temp.py
Adiabatic flame temperature T_ad via the Newton-Raphson method.

Idea: for constant-pressure combustion with no heat exchange, the enthalpy
of the reactants (at T_in) equals the enthalpy of the products (at T_ad):

    H_react(T_in) = H_prod(T_ad)

We define a residual function f(T) = H_prod(T) - H_react(T_in) = 0 and
search for its root. The derivative f'(T) = sum(n_i * cp_i(T)) is simply
the heat capacity of the products -> hence the Newton method.
"""

import numpy as np
from thermo import h_molar, cp_molar, stoich_moles


def H_mixture(moles, T):
    """Total mixture enthalpy [J] for a dict of moles."""
    return sum(n * h_molar(sp, T) for sp, n in moles.items())


def Cp_mixture(moles, T):
    """Total mixture heat capacity [J/K]."""
    return sum(n * cp_molar(sp, T) for sp, n in moles.items())


def adiabatic_T(phi, T_in=298.15, tol=1e-6, max_iter=100, T0=2000.0):
    """
    Compute the adiabatic flame temperature.

    Parameters:
        phi    - equivalence ratio
        T_in   - initial reactant temperature [K]
        tol    - tolerance on |f(T)| [J]
        T0     - iteration starting point [K]

    Returns:
        (T_ad, iteration_history_list)
    """
    reactants, products = stoich_moles(phi)
    H_react = H_mixture(reactants, T_in)

    T = T0
    history = []
    for k in range(max_iter):
        f = H_mixture(products, T) - H_react
        df = Cp_mixture(products, T)          # f'(T) = Cp_prod(T)
        history.append((k, T, f))
        if abs(f) < tol * max(1.0, abs(H_react)):
            break
        T_new = T - f / df
        # limit the step for stability
        T_new = min(max(T_new, 300.0), 4000.0)
        if abs(T_new - T) < 1e-8:
            T = T_new
            break
        T = T_new
    return T, history


def sweep_phi(phi_array, T_in=298.15):
    """Return T_ad for an array of phi values."""
    return np.array([adiabatic_T(phi, T_in)[0] for phi in phi_array])


if __name__ == "__main__":
    T_ad, hist = adiabatic_T(1.0)
    print(f"T_ad (phi=1) = {T_ad:.1f} K")
    print("Iterations:")
    for k, T, f in hist:
        print(f"  {k:2d}  T={T:8.2f} K   f={f: .3e} J")
