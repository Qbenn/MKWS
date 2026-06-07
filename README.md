# Numerical Methods for Modeling Methane–Air Combustion

Project for the course **Computational Methods in Combustion (MKWS)**, Warsaw
University of Technology, Faculty of Power and Aeronautical Engineering.

A from-scratch implementation of three classes of numerical methods applied to
the modeling of methane (CH₄) combustion. Ready-made kinetics packages
(e.g. Cantera) were deliberately avoided in order to implement and verify the
numerical methods themselves.

## Scope

| Part | Problem | Numerical method |
|------|---------|------------------|
| 1 | Adiabatic flame temperature *T*<sub>ad</sub>(φ) | Newton–Raphson (nonlinear equation) |
| 2 | Ignition in a 0D reactor, *T*(*t*), *Y*(*t*), Arrhenius plot | 4th-order Runge–Kutta (ODE system) |
| 3 | 1D laminar flame profile | finite differences + Newton + Thomas algorithm (PDE) |

## File structure

```
projekt_spalanie_mkws/
├── thermo.py          # NASA data, mixture properties, stoichiometry
├── flame_temp.py      # T_ad via Newton–Raphson
├── reactor0d.py       # 0D reactor, Arrhenius kinetics, RK4 integration
├── flame1d.py         # 1D laminar flame, finite differences, Thomas algorithm
├── main.py            # runs all parts and generates the figures
├── figures/           # generated figures (PNG) used in the report
├── raport.tex         # LaTeX report source
├── raport.pdf         # compiled report
├── requirements.txt   # Python dependencies
└── README.md
```

## Requirements

- Python 3.8+
- libraries listed in `requirements.txt` (NumPy, Matplotlib)
- to compile the report: a LaTeX distribution with the `babel-polish` package
  (e.g. TeX Live) or an Overleaf account

## Usage

### Computational part

```bash
pip install -r requirements.txt
python3 main.py
```

The script runs all three parts and saves the figures to the `figures/`
directory. Key results are printed to the console, including *T*<sub>ad</sub>,
the ignition delay, and the flame thickness.

### Compiling the report

```bash
pdflatex raport.tex
pdflatex raport.tex
```

(run twice — the table of contents is generated on the second pass).
Alternatively, upload `raport.tex` together with the `figures/` directory to
Overleaf.

## Main results

- **Part 1:** *T*<sub>ad</sub> ≈ 2326 K for the stoichiometric mixture (φ = 1);
  the Newton method converges quadratically in 3–4 iterations.
- **Part 2:** the characteristic autoignition behavior was reproduced, along
  with the Arrhenius-type dependence of the ignition delay on temperature
  (apparent activation energy ≈ 175 kJ/mol).
- **Part 3:** a smooth temperature profile with a well-defined reaction zone;
  grid convergence of the solution was demonstrated.

## Model limitations

A single-step global kinetics model was used and product dissociation was
neglected, which overestimates the final temperatures relative to measured
values. Natural extensions include using a multi-step reaction mechanism
(e.g. GRI-Mech 3.0), an adaptive time step for the stiff kinetic system, and
determining the laminar burning velocity *S*<sub>u</sub> as an eigenvalue
problem.

> **Note.** The report (`raport.tex` / `raport.pdf`) is written in Polish, as
> required by the course. The source code, comments, and this README are in
> English.
