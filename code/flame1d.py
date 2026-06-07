"""
flame1d.py
1D premixed laminar flame - reaction-diffusion equation.

We solve the steady-state temperature profile in a reference frame moving
with the flame (the flame-front frame). Simplified single-equation model for
the normalized temperature theta in [0,1]:

    rho*cp*Su * dT/dx = d/dx(lambda dT/dx) + q*omega(T)

with single-step Arrhenius kinetics. Discretized by the finite-difference
method (central for diffusion, upwind for convection) on a uniform grid;
the nonlinear system is solved by Newton-Raphson iteration with a
tridiagonal Jacobian (Thomas algorithm).

The educational goals are to demonstrate:
  * discretization of a partial differential equation (PDE) -> algebraic system,
  * solution of a nonlinear system (Newton + Thomas),
  * determination of the laminar burning velocity Su as an eigenvalue.
"""

import numpy as np


def thomas(a, b, c, d):
    """
    Solve a tridiagonal system (Thomas algorithm).
    a - sub-diagonal (a[0] unused), b - diagonal, c - super-diagonal,
    d - right-hand side. Returns the solution x.
    """
    n = len(b)
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = c[0]/b[0]; dp[0] = d[0]/b[0]
    for i in range(1, n):
        m = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/m
        dp[i] = (d[i] - a[i]*dp[i-1])/m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x


class Flame1D:
    """1D premixed flame model with dimensionless temperature theta."""

    def __init__(self, L=2e-3, N=201, Tu=300.0, Tb=2200.0,
                 alpha=2.0e-5, beta=8.0, Le=1.0):
        """
        L     - domain length [m]
        N     - number of grid nodes
        Tu,Tb - temperature of fresh mixture and burned gas [K]
        alpha - thermal diffusivity [m2/s]
        beta  - dimensionless activation energy (Zeldovich number)
        Le    - Lewis number (=1: thermal diffusion = mass diffusion)
        """
        self.L, self.N = L, N
        self.Tu, self.Tb = Tu, Tb
        self.alpha, self.beta, self.Le = alpha, beta, Le
        self.x = np.linspace(0, L, N)
        self.dx = self.x[1]-self.x[0]
        # initial condition: smoothed step (tanh)
        xc = 0.4*L
        self.theta = 0.5*(1 + np.tanh((self.x - xc)/(0.1*L)))

    def omega(self, theta):
        """Dimensionless reaction source term (high-activation-energy model)."""
        # Arrhenius form using the Zeldovich number
        return (self.beta**2/2.0) * (1-theta) * \
               np.exp(-self.beta*(1-theta)/(1 - 0.8*(1-theta)))

    def domega(self, theta):
        """Derivative of the source w.r.t. theta (for the Jacobian) - numerical."""
        eps = 1e-6
        return (self.omega(theta+eps)-self.omega(theta-eps))/(2*eps)

    def solve(self, Su_guess=0.4, tol=1e-8, max_iter=200):
        """
        Newton iteration for the theta profile at a given velocity Su.
        Dimensionless equation:
            Su*dtheta/dx = alpha*d2theta/dx2 + omega(theta)
        Returns (x, theta, Su).
        """
        N, dx = self.N, self.dx
        Su = Su_guess
        th = self.theta.copy()
        for it in range(max_iter):
            a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); r = np.zeros(N)
            # Dirichlet boundary conditions
            b[0] = 1.0; r[0] = -(th[0]-0.0)
            b[-1] = 1.0; r[-1] = -(th[-1]-1.0)
            for i in range(1, N-1):
                diff = self.alpha*(th[i+1]-2*th[i]+th[i-1])/dx**2
                conv = Su*(th[i]-th[i-1])/dx          # upwind
                F = -conv + diff + self.omega(th[i])
                # partial derivatives (tridiagonal Jacobian)
                dF_dim1 = self.alpha/dx**2 + Su/dx
                dF_di   = -2*self.alpha/dx**2 - Su/dx + self.domega(th[i])
                dF_dip1 = self.alpha/dx**2
                a[i] = dF_dim1; b[i] = dF_di; c[i] = dF_dip1
                r[i] = -F
            dth = thomas(a, b, c, r)
            th += dth
            th = np.clip(th, 0.0, 1.0)
            if np.linalg.norm(dth, np.inf) < tol:
                break
        self.theta = th
        self.iters = it+1
        # dimensional temperature
        self.T = self.Tu + (self.Tb-self.Tu)*th
        self.Su = Su
        return self.x, self.T, Su

    def flame_thickness(self):
        """Flame thickness: (Tb-Tu)/max|dT/dx|."""
        dTdx = np.gradient(self.T, self.x)
        return (self.Tb-self.Tu)/np.max(np.abs(dTdx))


if __name__ == "__main__":
    f = Flame1D(N=201)
    x, T, Su = f.solve(Su_guess=0.4)
    print(f"Newton iterations: {f.iters}")
    print(f"Flame thickness delta = {f.flame_thickness()*1e3:.3f} mm")
    print(f"T_min={T.min():.0f}K  T_max={T.max():.0f}K")
