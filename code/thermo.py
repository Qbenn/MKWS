"""
thermo.py
Thermodynamic data and helper functions for methane-air combustion.

7-coefficient NASA polynomials are used to compute the specific heat cp(T)
and enthalpy h(T) of individual species.

Global (stoichiometric) reaction:
    CH4 + 2 (O2 + 3.76 N2) -> CO2 + 2 H2O + 7.52 N2
"""

import numpy as np

# Universal gas constant [J/(mol*K)]
R_UNIV = 8.314462618

# Molar masses [kg/mol]
M = {
    "CH4": 0.0160425,
    "O2":  0.0319988,
    "N2":  0.0280134,
    "CO2": 0.0440095,
    "H2O": 0.0180153,
}

# NASA-7 coefficients (CHEMKIN format).
# Each entry: (T_low, T_mid, T_high, coeffs_low[7], coeffs_high[7])
# cp/R = a1 + a2 T + a3 T^2 + a4 T^3 + a5 T^4
# h/RT = a1 + a2/2 T + a3/3 T^2 + a4/4 T^3 + a5/5 T^4 + a6/T
NASA = {
    "CH4": (200.0, 1000.0, 3500.0,
        [5.14987613e+00, -1.36709788e-02, 4.91800599e-05, -4.84743026e-08,
         1.66693956e-11, -1.02466476e+04, -4.64130376e+00],
        [7.48514950e-02, 1.33909467e-02, -5.73285809e-06, 1.22292535e-09,
         -1.01815230e-13, -9.46834459e+03, 1.84373180e+01]),
    "O2": (200.0, 1000.0, 3500.0,
        [3.78245636e+00, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09,
         3.24372837e-12, -1.06394356e+03, 3.65767573e+00],
        [3.28253784e+00, 1.48308754e-03, -7.57966669e-07, 2.09470555e-10,
         -2.16717794e-14, -1.08845772e+03, 5.45323129e+00]),
    "N2": (200.0, 1000.0, 3500.0,
        [3.53100528e+00, -1.23660988e-04, -5.02999433e-07, 2.43530612e-09,
         -1.40881235e-12, -1.04697628e+03, 2.96747038e+00],
        [2.95257637e+00, 1.39690040e-03, -4.92631603e-07, 7.86010195e-11,
         -4.60755204e-15, -9.23948688e+02, 5.87188762e+00]),
    "CO2": (200.0, 1000.0, 3500.0,
        [2.35677352e+00, 8.98459677e-03, -7.12356269e-06, 2.45919022e-09,
         -1.43699548e-13, -4.83719697e+04, 9.90105222e+00],
        [3.85746029e+00, 4.41437026e-03, -2.21481404e-06, 5.23490188e-10,
         -4.72084164e-14, -4.87591660e+04, 2.27163806e+00]),
    "H2O": (200.0, 1000.0, 3500.0,
        [4.19864056e+00, -2.03643410e-03, 6.52040211e-06, -5.48797062e-09,
         1.77197817e-12, -3.02937267e+04, -8.49032208e-01],
        [3.03399249e+00, 2.17691804e-03, -1.64072518e-07, -9.70419870e-11,
         1.68200992e-14, -3.00042971e+04, 4.96677010e+00]),
}


def _coeffs(species, T):
    """Return the appropriate NASA coefficient set for a given temperature."""
    T_low, T_mid, T_high, c_low, c_high = NASA[species]
    Tc = min(max(T, T_low), T_high)
    return np.array(c_low if Tc < T_mid else c_high)


def cp_molar(species, T):
    """Molar specific heat [J/(mol*K)]."""
    a = _coeffs(species, T)
    cp_over_R = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
    return cp_over_R * R_UNIV


def h_molar(species, T):
    """Molar enthalpy (including enthalpy of formation) [J/mol]."""
    a = _coeffs(species, T)
    h_over_RT = (a[0] + a[1]/2*T + a[2]/3*T**2 + a[3]/4*T**3
                 + a[4]/5*T**4 + a[5]/T)
    return h_over_RT * R_UNIV * T


def cp_mass(species, T):
    """Mass-based specific heat [J/(kg*K)]."""
    return cp_molar(species, T) / M[species]


def stoich_moles(phi):
    """
    Mixture composition per 1 mole of CH4 fuel for a given equivalence ratio phi.
    Oxidation: CH4 + 2/phi (O2 + 3.76 N2).
    Returns dicts of reactant and product moles (complete combustion, phi<=1).
    """
    a = 2.0 / phi          # moles of O2
    n_N2 = a * 3.76
    reactants = {"CH4": 1.0, "O2": a, "N2": n_N2}
    # products (for phi<=1: excess O2 remains)
    products = {"CO2": 1.0, "H2O": 2.0, "N2": n_N2, "O2": max(a - 2.0, 0.0)}
    return reactants, products


if __name__ == "__main__":
    # quick test
    print("cp CO2 @ 1000K =", cp_molar("CO2", 1000.0), "J/mol/K")
    print("h H2O @ 298K   =", h_molar("H2O", 298.15), "J/mol")
