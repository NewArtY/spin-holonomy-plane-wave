# -*- coding: utf-8 -*-
r"""
spin_magnitude.py -- how big is the NET spin rotation of an electron crossing a
finite plane-wave laser pulse, and how much of it is controlled by the CEP?
(Settles plan/05-physics-audit.md, section 8.)

This script deliberately does NOT reuse bench/physics_np.py: that module encodes
the toy model of the original draft, in which the vector potential is a function
of PROPER TIME, A_mu(tau).  That is not a solution of Maxwell's equations.  Here
everything is a function of the light-cone phase

    eta = k.x = omega (t - z/c),        k^mu = (omega/c)(1,0,0,1),  k.k = 0.

================================================================================
1. FIELD
================================================================================
Normalised transverse potential (dimensionless; q = electron charge;
a0 = |q| E0/(m c omega) is the usual intensity parameter):

    a^mu(eta) = q A^mu/(m c^2) = (0, a_x(eta), a_y(eta), 0)

    a_x = a0 f(eta) cos(eta+phi0)/sqrt(1+d^2)
    a_y = a0 d f(eta) sin(eta+phi0)/sqrt(1+d^2)          d = ellipticity
                                                         d=0 linear, d=1 circular

The pulse is specified through the POTENTIAL, not through the field.  Reason: the
only condition Maxwell + a finite pulse actually impose is that the transverse
field carries no DC component,

    int E deta = 0     <=>     a(+inf) = a(-inf) = 0,

which is automatic for any envelope with f(+-inf)=0.  If instead one writes
E = E0 f(eta) cos(eta+phi0) and integrates, one gets
a(+inf)-a(-inf) = -a0 sigma sqrt(2pi) exp(-sigma^2/2) cos(phi0) != 0: a net
momentum kick, an electron that never returns to its initial 4-velocity, and a
spurious "CEP effect" that is a pure artefact of the parametrisation.

The extra condition int a deta = 0 that is sometimes quoted is NOT required by
Maxwell -- it fixes the net transverse DISPLACEMENT, not the field.  With the
potential-first convention int a_x deta = a0 sigma sqrt(2pi) e^{-sigma^2/2}
cos(phi0), which is precisely the xi_CEP spectral weight of the audit; forcing it
to zero would delete the leading CEP-odd term by hand.  Both conventions are
implemented (--zero-mean-a) and compared in stage t3(f); the difference is
numerically irrelevant (mu <= e^{-sigma^2/2}).

Envelopes (N = number of optical cycles inside the FWHM of the INTENSITY
envelope f^2, so that the two envelopes are compared at equal N):
    gauss :  f = exp(-eta^2/(2 sigma^2)),          sigma = pi N/sqrt(ln2) = 3.7745 N
             truncated at |eta| = 8 sigma  (f ~ 1.3e-14)
    cos2  :  f = cos^2(pi eta/(2L)) on |eta|<=L,   L = 8.6327 N  (compact support:
             the return condition a(+-inf)=0 then holds to machine precision)

================================================================================
2. ORBIT: exact Volkov solution (algebraic, nothing to integrate)
================================================================================
With u^mu = (gamma, gamma*beta), u.u = 1, metric (+,-,-,-),

    du^mu/deta = n^mu (a'.u) - a'^mu (n.u),   n^mu = k^mu/(k.u_0),  n.u = 1,

hence, with lam = u_0^0 - u_0^3 conserved,

    u_perp(eta) = u_{0,perp} - a_perp(eta)
    u^0 = [lam + (1+u_perp^2)/lam]/2 ,   u^3 = [(1+u_perp^2)/lam - lam]/2 .

Head-on geometry: wave along +z, electron along -z, lam = gamma(1+beta) ~ 2 gamma.

================================================================================
3. SPIN: Thomas-BMT, covariant form (Jackson 3rd ed. Eq. (11.170))
================================================================================
    dS^al/dtau = (q/mc) [ (g/2) F^{al be} S_be
                          + (g/2 - 1)(1/c^2) u^al (S_l F^{l be} u_be) ],
    a_anom = (g-2)/2 = g/2 - 1.

The factor (g-1) used in the original draft (bench/physics_np.py line 41) appears
nowhere in T-BMT; the rest-frame form has (a+1/gamma) and (a+1/(gamma+1)):
    dS/dt = Om x S,
    Om = -(q/mc)[(a+1/g)B - (a g/(g+1))(b.B)b - (a+1/(g+1)) b x E].
Reducing the covariant form to eta with the same normalisation as the orbit:

    dS^mu/deta = (1+a) [ n^mu (a'.S) - a'^mu (n.S) ]
                 + a  u^mu [ (n.S)(a'.u) - (a'.S) ]                        (*)

i.e.  dS/deta = (M + a P_u M) S,  M^{mu nu} = n^mu a'^nu - a'^mu n^nu in so(1,3),
P_u = projector orthogonal to u.  Verified analytically and numerically: (*)
conserves S.u and S.S exactly for any a.  At a=0 it is *identical* to the orbit
equation (the audit's point).

--------------------------------------------------------------------------------
3a. EXACT CLOSED FORM AND THE NULL THEOREM  (derived here, checked in stage t1)
--------------------------------------------------------------------------------
Write a'^mu = alpha_x(eta) eps_x^mu + alpha_y(eta) eps_y^mu, and
K_i = n ^ eps_i.  Using n.eps_i = 0, eps_i.eps_j = -delta_ij, n.n = 0 one gets
K_x K_y = K_y K_x = 0, hence [K_x,K_y] = 0 and

    M(eta) = alpha_x K_x + alpha_y K_y ,   [M(eta1), M(eta2)] = 0 ,
    Lambda(eta) = exp[a_x(eta) K_x + a_y(eta) K_y]   (the g=2 propagator).

Because Lambda commutes with M and P_u = Lambda P_{u_0} Lambda^{-1}, the
interaction-picture generator is

    Lambda^{-1} (P_u M) Lambda = P_{u_0} M(eta)  =  alpha_x (P K_x) + alpha_y (P K_y)

with P = P_{u_0} CONSTANT.  Consequences:

  (i) LINEAR polarisation (alpha_y = 0).  The generator is a fixed matrix times a
      scalar function, so the T-ordering is vacuous and the exact solution is

          S(eta) = exp[a_x(eta) K_x] . exp[a_anom * a_x(eta) * P K_x] . S(-inf)

      Since a_x(+inf) = a_x(-inf) = 0, BOTH factors are the identity at the end:

          *** S(+inf) = S(-inf) EXACTLY, FOR ANY g, ANY ENVELOPE, ANY CEP. ***

      The net rotation vanishes not only at g=2 (as the audit says) but for the
      real electron as well.  There is no O(a_anom) effect and no O(a_anom gamma)
      effect: the whole eta-dynamics, written in the initial rest frame, does not
      contain gamma at all (a boost along z leaves a^mu=(0,a_x,a_y,0) invariant
      and rescales n so that n.u_0 = 1).

 (ii) ELLIPTICAL polarisation.  [P K_x, P K_y] != 0, so the T-ordering survives.
      The first Magnus term still vanishes (int alpha_i deta = 0), so the net
      rotation starts at SECOND order in the anomaly:
          Omega_2 = (a_anom^2/2) [P K_x, P K_y] * Area,
          Area = int (a_x alpha_y - a_y alpha_x) deta   (area swept in the a-plane)
      -> Theta_net ~ a_anom^2 a0^2 N, i.e. ~1e-6 times a geometric factor.

================================================================================
4. OBSERVABLE
================================================================================
u(+inf) = u(-inf) exactly, so S(-inf) -> S(+inf) restricted to the 3-space
orthogonal to u_0 is an element of SO(3).  We propagate the orthonormal
rest-frame triad e_1 = x, e_2 = y, e_3 = boosted z and read off

    R_{ji} = -eta_{mu nu} e_j^mu(-inf) e_i^nu(+inf)     (Euclidean = -Minkowski)
    Theta  = |rotvec(R)|          net rotation angle (basis-free, invariant)
    DSigma = rotvec(R)_y          signed rotation about the normal of the
                                  polarisation plane -- the paper's Sigma
    P_flip = sin^2(Theta/2)       max over initial spin directions, spin-1/2

================================================================================
5. STAGES
================================================================================
    t1  validation: the g = 2 null, the stronger null for linear polarisation at
        inflated anomaly, the exact closed form, and the invariants
    t2  main grid: net angle and CEP-dependent part, linear and circular
    t3  scaling laws in gamma, a0, the anomaly and the CEP weight
    t4  antisymmetry of the in-pulse rotation under phi0 -> phi0 + pi
    t5  applicability window (closed-form estimates only, nothing is simulated)
    t6  convergence in the integrator tolerance -- TABLE S3 of the Supplemental
        Material; added 2026-08-02, the table used to have no script behind it
    t7  the longitudinal spin projection Delta S_par and its exponent in the
        anomaly -- the numerical support for the O(a_e^6) helicity statement of
        Sec. III.E; added 2026-08-02, nothing computed it before

NOTATION.  The anomaly is a_e = (g-2)/2, spelled `anom` in the source and `no`
in the manuscript.  It is NOT to be confused with a_x, a_y, a_perp (the
normalised transverse potential) or a0 (its amplitude).  Earlier drafts of the
manuscript wrote the anomaly as plain `a`, which is why some of the derivations
in this docstring still use `a_anom` for it.

Usage
    python spin_magnitude.py --stages t1 --out sm_t1.json
    python spin_magnitude.py --stages t6,t7 --out sm_t67.json
    python spin_magnitude.py --merge sm_t1.json,sm_t2.json --out spin_magnitude.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time

import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

# ---------------------------------------------------------------- constants
ANOM = 1.15965218059e-3          # a = (g-2)/2, CODATA 2022 electron
ALPHA = 7.2973525693e-3          # fine-structure constant
MC2_EV = 510998.95               # m_e c^2 [eV]
HBAR_OMEGA_EV = 1.55             # Ti:sapphire, 800 nm
SEED = 20260730

SIG_PER_N = math.pi / math.sqrt(math.log(2.0))            # 3.7745142
_X_HALF = math.acos(2.0 ** -0.25)                         # 0.5716701
L_PER_N = math.pi ** 2 / (2.0 * _X_HALF)                  # 8.6326733
N_SIGMA_TRUNC = 8.0

ETA = np.diag([1.0, -1.0, -1.0, -1.0])


# ---------------------------------------------------------------- field
class Pulse:
    """Transverse normalised potential (a_x, a_y)(eta) and its derivative."""

    def __init__(self, a0=1.0, N=1.0, phi0=0.0, env="gauss", delta=0.0,
                 sigma=None, zero_mean_a=False):
        self.a0 = float(a0)
        if N is None:
            if sigma is None:
                raise ValueError("give N or sigma")
            N = float(sigma) / SIG_PER_N
        self.N = float(N)
        self.phi0 = float(phi0)
        self.env = env
        self.delta = float(delta)
        self.zero_mean_a = bool(zero_mean_a)
        self.norm = 1.0 / math.sqrt(1.0 + self.delta ** 2)
        if env == "gauss":
            self.sigma = float(sigma) if sigma is not None else SIG_PER_N * self.N
            self.L = None
            self.eta_max = N_SIGMA_TRUNC * self.sigma
            self.sigma_eff = self.sigma
        elif env == "cos2":
            self.L = L_PER_N * self.N
            self.sigma = None
            self.eta_max = self.L
            self.sigma_eff = SIG_PER_N * self.N
        else:
            raise ValueError(env)
        self.mux, self.muy = self._mean_correction() if self.zero_mean_a else (0.0, 0.0)

    # --- envelope ---------------------------------------------------------
    def f(self, eta):
        if self.env == "gauss":
            return np.exp(-eta ** 2 / (2.0 * self.sigma ** 2))
        w = math.pi / (2.0 * self.L)
        return np.where(np.abs(eta) <= self.L, np.cos(w * eta) ** 2, 0.0)

    def df(self, eta):
        if self.env == "gauss":
            return -eta / self.sigma ** 2 * self.f(eta)
        w = math.pi / (2.0 * self.L)
        return np.where(np.abs(eta) <= self.L,
                        -2.0 * w * np.cos(w * eta) * np.sin(w * eta), 0.0)

    def _mean_correction(self):
        """(mux, muy) such that int (carrier - mu) f deta = 0 for each component."""
        if self.env == "gauss":
            e = math.exp(-self.sigma ** 2 / 2.0)
            return e * math.cos(self.phi0), self.delta * e * math.sin(self.phi0)
        den = quad(lambda e: float(self.f(e)), -self.L, self.L, limit=400)[0]
        nx = quad(lambda e: float(self.f(e)) * math.cos(e + self.phi0),
                  -self.L, self.L, limit=400)[0]
        ny = quad(lambda e: float(self.f(e)) * math.sin(e + self.phi0),
                  -self.L, self.L, limit=400)[0]
        return nx / den, self.delta * ny / den

    # --- potential --------------------------------------------------------
    def a(self, eta):
        f = self.f(eta)
        s = self.a0 * self.norm
        ax = s * f * (np.cos(eta + self.phi0) - self.mux)
        ay = s * self.delta * f * (np.sin(eta + self.phi0)) - s * f * self.muy
        return ax, ay

    def da(self, eta):
        f, df = self.f(eta), self.df(eta)
        c, sn = np.cos(eta + self.phi0), np.sin(eta + self.phi0)
        s = self.a0 * self.norm
        dax = s * (df * (c - self.mux) - f * sn)
        day = s * self.delta * (df * sn + f * c) - s * self.muy * df
        return dax, day

    # --- diagnostics ------------------------------------------------------
    def integral_a(self):
        ix = quad(lambda e: float(self.a(e)[0]), -self.eta_max, self.eta_max,
                  limit=500)[0]
        iy = quad(lambda e: float(self.a(e)[1]), -self.eta_max, self.eta_max,
                  limit=500)[0]
        return ix, iy

    def endpoint_a(self):
        lo, hi = self.a(-self.eta_max), self.a(self.eta_max)
        return float(max(abs(lo[0]), abs(lo[1]), abs(hi[0]), abs(hi[1])))

    def swept_area(self):
        """int (a_x a_y' - a_y a_x') deta -- the source of the elliptical effect."""
        def integ(e):
            ax, ay = self.a(e)
            dx, dy = self.da(e)
            return float(ax * dy - ay * dx)
        return quad(integ, -self.eta_max, self.eta_max, limit=800)[0]

    def as_dict(self):
        return dict(a0=self.a0, N=self.N, phi0=self.phi0, env=self.env,
                    delta=self.delta, sigma_eff=self.sigma_eff, L=self.L,
                    eta_max=self.eta_max, zero_mean_a=self.zero_mean_a,
                    mux=self.mux, muy=self.muy)


# ---------------------------------------------------------------- kinematics
def mink(v, w):
    return v[..., 0] * w[..., 0] - (v[..., 1] * w[..., 1]
                                    + v[..., 2] * w[..., 2] + v[..., 3] * w[..., 3])


def u_initial(gamma, head_on=True):
    b = math.sqrt(max(gamma ** 2 - 1.0, 0.0))
    return np.array([gamma, 0.0, 0.0, -b if head_on else b])


def triad(u0):
    g, uz = u0[0], u0[3]
    return np.stack([np.array([0.0, 1.0, 0.0, 0.0]),
                     np.array([0.0, 0.0, 1.0, 0.0]),
                     np.array([uz, 0.0, 0.0, g])])


def u_of_eta(eta, pulse, u0, lam):
    ax, ay = pulse.a(eta)
    ux, uy = u0[1] - ax, u0[2] - ay
    q = (1.0 + ux * ux + uy * uy) / lam
    return np.array([0.5 * (lam + q), ux, uy, 0.5 * (q - lam)])


# ---------------------------------------------------------------- spin ODE
def make_rhs(pulse, u0, lam, anom):
    inv = 1.0 / lam
    c1 = 1.0 + anom

    def rhs(eta, y):
        ax, ay = pulse.a(eta)
        dax, day = pulse.da(eta)
        ux, uy = u0[1] - ax, u0[2] - ay
        q = (1.0 + ux * ux + uy * uy) / lam
        u = np.array([0.5 * (lam + q), ux, uy, 0.5 * (q - lam)])
        S = y.reshape(-1, 4)
        nS = (S[:, 0] - S[:, 3]) * inv                    # n.S
        aS = -(dax * S[:, 1] + day * S[:, 2])             # a'.S
        au = -(dax * ux + day * uy)                       # a'.u
        w = anom * (nS * au - aS)
        dS = np.empty_like(S)
        dS[:, 0] = c1 * aS * inv + w * u[0]
        dS[:, 1] = -c1 * dax * nS + w * u[1]
        dS[:, 2] = -c1 * day * nS + w * u[2]
        dS[:, 3] = c1 * aS * inv + w * u[3]
        return dS.ravel()

    return rhs


def rotvec_from_matrix(R):
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = float(np.linalg.norm(w))
    c = 0.5 * (float(np.trace(R)) - 1.0)
    th = math.atan2(s, c)
    if s < 1e-15:
        return w if c > 0 else np.array([np.nan, np.nan, np.nan])
    return w * (th / s)


def net_rotation(gamma, pulse, anom=ANOM, rtol=1e-12, atol=1e-14, head_on=True,
                 n_track=0, with_matrix=False):
    """Integrate the rest-frame triad through the pulse.

    with_matrix adds the raw 3x3 SO(3) matrix R to the returned dictionary.  R
    acts on COMPONENTS in the initial rest-frame triad (e_1, e_2, e_3), whose
    third leg is the boosted propagation direction nhat, so R[2][2] - 1 is the
    change in the longitudinal spin projection of a spin that starts along nhat
    -- the quantity the helicity statement of the main text is about.
    """
    u0 = u_initial(gamma, head_on)
    lam = u0[0] - u0[3]
    E0 = triad(u0)
    rhs = make_rhs(pulse, u0, lam, anom)
    t0, t1 = -pulse.eta_max, +pulse.eta_max
    te = np.linspace(t0, t1, n_track) if n_track else None
    sol = solve_ivp(rhs, (t0, t1), E0.ravel(), method="DOP853",
                    rtol=rtol, atol=atol, t_eval=te)
    if not sol.success:
        raise RuntimeError(sol.message)
    Ef = sol.y[:, -1].reshape(3, 4)
    R = np.array([[-mink(E0[j], Ef[i]) for i in range(3)] for j in range(3)])
    rv = rotvec_from_matrix(R)
    uf = u_of_eta(t1, pulse, u0, lam)
    out = dict(theta=float(np.linalg.norm(rv)),
               rx=float(rv[0]), ry=float(rv[1]), rz=float(rv[2]),
               orth_err=float(np.max(np.abs(R.T @ R - np.eye(3)))),
               nfev=int(sol.nfev),
               du=float(np.max(np.abs(uf - u0))),
               su_err=float(np.max(np.abs([mink(Ef[i], uf) for i in range(3)]))),
               ss_err=float(np.max(np.abs([mink(Ef[i], Ef[i]) + 1.0
                                           for i in range(3)]))))
    if with_matrix:
        out["R"] = R.tolist()
    if n_track:
        tr = [float(np.linalg.norm(rotvec_from_matrix(R)))
              for R in _profile_from_sol(sol, pulse, u0, lam, E0)]
        out["eta_track"] = sol.t.tolist()
        out["theta_track"] = tr
        out["theta_transient_max"] = float(np.max(tr))
    return out


# ------------------------------------------------- in-pulse (Wigner) rotation
def boost_matrix(u0, u):
    """Pure boost B with B u0 = u, as a mixed-index 4x4 matrix."""
    g = 1.0 + mink(u, u0)
    s = u + u0
    return np.eye(4) - np.outer(s, s @ ETA) / g + 2.0 * np.outer(u, u0 @ ETA)


def _profile_from_sol(sol, pulse, u0, lam, E0):
    """Rotation of the transported triad relative to the PURELY BOOSTED triad.

    This is the accumulated Wigner/Thomas rotation at phase eta -- the only
    frame-independent way to speak about the spin orientation *inside* the pulse,
    where u(eta) != u_0.
    """
    out = []
    for m in range(sol.y.shape[1]):
        Em = sol.y[:, m].reshape(3, 4)
        um = u_of_eta(sol.t[m], pulse, u0, lam)
        B = boost_matrix(u0, um)
        Ref = np.array([B @ E0[j] for j in range(3)])
        out.append(np.array([[-mink(Ref[j], Em[i]) for i in range(3)]
                             for j in range(3)]))
    return out


def rotation_profile(gamma, pulse, anom, etas, rtol=1e-13, atol=1e-16):
    """List of 3x3 Wigner-rotation matrices at the requested phases."""
    u0 = u_initial(gamma)
    lam = u0[0] - u0[3]
    E0 = triad(u0)
    rhs = make_rhs(pulse, u0, lam, anom)
    etas = np.asarray(etas, float)
    sol = solve_ivp(rhs, (-pulse.eta_max, float(etas[-1])), E0.ravel(),
                    method="DOP853", rtol=rtol, atol=atol, t_eval=etas)
    if not sol.success:
        raise RuntimeError(sol.message)
    return _profile_from_sol(sol, pulse, u0, lam, E0)


# ---------------------------------------------------------------- closed form
def closed_form_linear(gamma, pulse, anom, eta):
    """S(eta) = exp[a_x K_x] exp[anom a_x P K_x] S(-inf)  (linear polarisation)."""
    u0 = u_initial(gamma)
    lam = u0[0] - u0[3]
    n = np.array([1.0, 0.0, 0.0, 1.0]) / lam
    ex = np.array([0.0, 1.0, 0.0, 0.0])
    Kup = np.outer(n, ex) - np.outer(ex, n)            # K^{mu nu}
    K = Kup @ ETA                                      # K^mu_nu
    P = np.eye(4) - np.outer(u0, u0 @ ETA)             # P^mu_nu
    ax = float(pulse.a(eta)[0])
    return expm(ax * K) @ expm(anom * ax * (P @ K))


# ---------------------------------------------------------------- harmonics
def cep_harmonics(vals, nmax=4):
    v = np.asarray(vals, float)
    M = len(v)
    F = np.fft.rfft(v) / M
    out = {"mean": float(F[0].real), "ptp_half": float(0.5 * (v.max() - v.min()))}
    for n in range(1, min(nmax, M // 2) + 1):
        out[f"A{n}"] = float(2.0 * abs(F[n]))
    return out


def scan_phi0(gamma, a0, N, env="gauss", delta=0.0, n_phi=16, anom=ANOM,
              rtol=1e-12, atol=1e-14, sigma=None, zero_mean_a=False):
    phis = 2.0 * math.pi * np.arange(n_phi) / n_phi
    theta, rvy, rvz, diag = [], [], [], []
    for p in phis:
        pu = Pulse(a0=a0, N=N, phi0=float(p), env=env, delta=delta, sigma=sigma,
                   zero_mean_a=zero_mean_a)
        r = net_rotation(gamma, pu, anom=anom, rtol=rtol, atol=atol)
        theta.append(r["theta"])
        rvy.append(r["ry"])
        rvz.append(r["rz"])
        diag.append(r)
    hy = cep_harmonics(rvy)
    ht = cep_harmonics(theta)
    pu0 = Pulse(a0=a0, N=N, phi0=0.0, env=env, delta=delta, sigma=sigma)
    return dict(gamma=gamma, a0=a0, N=pu0.N, env=env, delta=delta,
                sigma_eff=pu0.sigma_eff, n_phi=n_phi, zero_mean_a=zero_mean_a,
                phi0=phis.tolist(), theta=theta, ry=rvy, rz=rvz,
                theta_mean=float(np.mean(theta)), theta_max=float(np.max(theta)),
                theta_cep_half=ht["ptp_half"], theta_A1=ht.get("A1"),
                theta_A2=ht.get("A2"),
                ry_mean=hy["mean"], ry_cep_half=hy["ptp_half"],
                ry_A1=hy.get("A1"), ry_A2=hy.get("A2"), ry_A3=hy.get("A3"),
                Pflip_max=float(math.sin(max(theta) / 2.0) ** 2),
                swept_area=pu0.swept_area(),
                noise=dict(orth=max(d["orth_err"] for d in diag),
                           du=max(d["du"] for d in diag),
                           su=max(d["su_err"] for d in diag),
                           ss=max(d["ss_err"] for d in diag),
                           nfev=max(d["nfev"] for d in diag)))


# ---------------------------------------------------------------- stages
def stage_t1(cfg):
    """Validation.

    (a) g = 2  -> net rotation must be 0 to integrator accuracy (audit's test).
    (b) ANY g (anomaly scanned up to 10, i.e. 4 orders above the physical value)
        -> still 0 for linear polarisation: the stronger null theorem of Sec. 3a.
    (c) exact closed form vs. the integrator (linear polarisation, mid-pulse).
    (d) invariants S.u, S.S, and u(+inf) = u(-inf).
    """
    t0 = time.perf_counter()
    g2, anyg = [], []
    envs = cfg["t1_envs"]
    gam1, a01, Ns1 = cfg["t1_gammas"], cfg["t1_a0s"], cfg["t1_Ns"]
    for env in envs:
        for gamma in gam1:
            for a0 in a01:
                for N in Ns1:
                    for phi0 in cfg["t1_phis"]:
                        pu = Pulse(a0=a0, N=N, phi0=phi0, env=env)
                        r = net_rotation(gamma, pu, anom=0.0, rtol=1e-13, atol=1e-16)
                        r.update(env=env, gamma=gamma, a0=a0, N=N, phi0=phi0,
                                 end_a=pu.endpoint_a())
                        g2.append(r)
    for env in envs:
        for gamma in gam1[::2]:
            for a0 in a01[-2:]:
                for A in cfg["t1_anoms"]:
                    pu = Pulse(a0=a0, N=1.0, phi0=0.3, env=env)
                    r = net_rotation(gamma, pu, anom=A, rtol=1e-13, atol=1e-16)
                    r.update(env=env, gamma=gamma, a0=a0, N=1.0, anom=A)
                    anyg.append(r)
    # (c) closed form
    cf = []
    for gamma in cfg["t1_cf_gammas"]:
        for a0 in a01[-2:]:
            pu = Pulse(a0=a0, N=1.0, phi0=0.3, env="cos2")
            u0 = u_initial(gamma)
            E0 = triad(u0)
            for frac in (0.25, 0.5, 0.75):
                eta = -pu.eta_max + frac * 2 * pu.eta_max
                rhs = make_rhs(pu, u0, u0[0] - u0[3], ANOM)
                sol = solve_ivp(rhs, (-pu.eta_max, eta), E0.ravel(),
                                method="DOP853", rtol=1e-13, atol=1e-16)
                num = sol.y[:, -1].reshape(3, 4)
                U = closed_form_linear(gamma, pu, ANOM, eta)
                ana = np.array([U @ E0[i] for i in range(3)])
                err = float(np.max(np.abs(num - ana)) / max(1.0, gamma))
                cf.append(dict(gamma=gamma, a0=a0, eta_frac=frac, rel_err=err))
    return dict(g2_cases=g2, anyg_cases=anyg, closed_form=cf,
                max_theta_g2=max(c["theta"] for c in g2),
                max_theta_anyg=max(c["theta"] for c in anyg),
                max_orth=max(c["orth_err"] for c in g2 + anyg),
                max_du=max(c["du"] for c in g2 + anyg),
                max_su=max(c["su_err"] for c in g2 + anyg),
                max_ss=max(c["ss_err"] for c in g2 + anyg),
                max_closed_form_err=max(c["rel_err"] for c in cf),
                wall_s=time.perf_counter() - t0)


def stage_t2(cfg):
    """Main grid: net angle and CEP-dependent part, linear and circular."""
    t0 = time.perf_counter()
    rows = []
    for delta in cfg["deltas"]:
        n_phi = cfg["n_phi_lin"] if delta == 0.0 else cfg["n_phi_cir"]
        for gamma in cfg["gammas"]:
            for a0 in cfg["a0s"]:
                for N in cfg["Ns"]:
                    r = scan_phi0(gamma, a0, N, env="gauss", delta=delta,
                                  n_phi=n_phi, rtol=cfg["rtol"], atol=cfg["atol"])
                    for k in ("phi0", "theta", "ry", "rz"):
                        r.pop(k)
                    rows.append(r)
    return dict(rows=rows, wall_s=time.perf_counter() - t0)


def stage_t3(cfg):
    """Scaling laws: gamma, a0, anomaly, xi_CEP."""
    t0 = time.perf_counter()
    res = {}
    keys = ("theta", "rx", "ry", "rz", "orth_err", "du")

    # (a) gamma scan -- tests the claimed a*gamma enhancement, linear + circular
    for delta in (0.0, 1.0):
        res[f"gamma_scan_d{delta:g}"] = [
            dict(gamma=g, **{k: net_rotation(
                g, Pulse(a0=5.0, N=1.0, phi0=0.0, delta=delta),
                rtol=1e-13, atol=1e-16)[k] for k in keys})
            for g in (1.0, 3.0, 10.0, 30.0, 1e2, 3e2, 1e3, 3e3, 1e4)]

    # (b) a0 scan, circular
    res["a0_scan_d1"] = [
        dict(a0=a, **{k: net_rotation(1e3, Pulse(a0=a, N=1.0, phi0=0.0, delta=1.0),
                                      rtol=1e-13, atol=1e-16)[k] for k in keys},
             area=Pulse(a0=a, N=1.0, delta=1.0).swept_area())
        for a in (0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0)]

    # (c) N scan, circular
    res["N_scan_d1"] = [
        dict(N=n, **{k: net_rotation(1e3, Pulse(a0=10.0, N=n, phi0=0.0, delta=1.0),
                                     rtol=1e-13, atol=1e-16)[k] for k in keys},
             area=Pulse(a0=10.0, N=n, delta=1.0).swept_area())
        for n in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]

    # (d) anomaly scan, circular -> expect Theta ~ anom^2
    res["anom_scan_d1"] = [
        dict(anom=A, **{k: net_rotation(1e3, Pulse(a0=10.0, N=4.0, phi0=0.0,
                                                   delta=1.0),
                                        anom=A, rtol=1e-13, atol=1e-16)[k]
                        for k in keys})
        for A in (ANOM, 3 * ANOM, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)]

    # (e) ellipticity scan
    res["delta_scan"] = [
        dict(delta=d, **{k: net_rotation(1e3, Pulse(a0=10.0, N=4.0, phi0=0.0,
                                                    delta=d),
                                         rtol=1e-13, atol=1e-16)[k] for k in keys},
             area=Pulse(a0=10.0, N=4.0, delta=d).swept_area())
        for d in (0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0)]

    # (f) CEP suppression vs sigma (sub-cycle -> multi-cycle), elliptical.
    #     Run at gamma = 1 (= initial rest frame): the dynamics is gamma-independent
    #     (proved in Sec. 3a, verified in (a)) and the lab-frame formulation loses
    #     ~gamma^2 * eps digits to cancellation, which would bury a 1e-12 signal.
    cep = []
    for s in cfg.get("t3_sigmas",
                     (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.7745, 5.0, 7.549)):
        for d in (0.5, 1.0):
            r = scan_phi0(1.0, 10.0, None, env="gauss", delta=d,
                          n_phi=cfg.get("t3_nphi", 32),
                          rtol=1e-13, atol=1e-16, sigma=s)
            for k in ("phi0", "theta", "ry", "rz"):
                r.pop(k)
            r.update(sigma=s, N_equiv=s / SIG_PER_N,
                     xi_CEP=math.exp(-s ** 2 / 2.0), xi2=math.exp(-2.0 * s ** 2))
            cep.append(r)
    res["cep_vs_sigma"] = cep

    # (g) convention / envelope control
    ctl = []
    for env in ("gauss", "cos2"):
        for zm in (False, True):
            r = scan_phi0(1.0, 10.0, 1.0, env=env, delta=1.0, n_phi=16,
                          rtol=1e-13, atol=1e-16, zero_mean_a=zm)
            for k in ("phi0", "theta", "ry", "rz"):
                r.pop(k)
            ctl.append(r)
    res["envelope_control"] = ctl

    # (h) transient (in-pulse) Wigner rotation -- what the draft actually plots.
    #     Reported: total, and the anomalous part R_a . R_{a=0}^{-1}.
    trans = []
    for phi0 in (0.0, math.pi / 4, math.pi / 2, math.pi):
        for a0 in (1.0, 10.0):
            pu = Pulse(a0=a0, N=1.0, phi0=phi0, env="gauss", delta=0.0)
            etas = np.linspace(-pu.eta_max, pu.eta_max, 129)
            Ra = rotation_profile(1.0, pu, ANOM, etas)
            R0 = rotation_profile(1.0, pu, 0.0, etas)
            tot = [float(np.linalg.norm(rotvec_from_matrix(R))) for R in Ra]
            ano = [float(rotvec_from_matrix(Ra[i] @ R0[i].T)[1])
                   for i in range(len(etas))]
            trans.append(dict(phi0=phi0, a0=a0,
                              theta_total_max=max(tot),
                              theta_total_end=tot[-1],
                              anom_rot_max=float(np.max(np.abs(ano))),
                              anom_rot_end=ano[-1],
                              a_x_peak=float(np.max(np.abs(pu.a(etas)[0])))))
    res["transient"] = trans
    res["wall_s"] = time.perf_counter() - t0
    return res


def stage_t4(cfg):
    """Antisymmetry  DSigma(phi0+pi) = -DSigma(phi0)?

    For the NET rotation in linear polarisation the question is vacuous (the net
    is identically zero), so the test is run on the two observables that are not
    identically zero:
      (i)  the IN-PULSE Wigner rotation at the pulse centre, total and anomalous
           part, linear polarisation -> tests audit Sec. 11 directly;
      (ii) the net rotation for circular polarisation.
    All at gamma = 1 (rest frame) for numerical cleanliness.
    """
    t0 = time.perf_counter()
    n_phi = cfg["t4_nphi"]
    phis = 2.0 * math.pi * np.arange(n_phi) / n_phi

    def split(v):
        v = np.asarray(v, float)
        even = 0.5 * (v + np.roll(v, n_phi // 2))
        return float(np.max(np.abs(even))), float(np.max(np.abs(v - even)))

    inpulse = []
    for a0 in cfg["t4_a0s"]:
        for s in cfg["t4_sigmas"]:
            tot, ano = [], []
            for p in phis:
                pu = Pulse(a0=a0, N=None, phi0=float(p), env="gauss",
                           delta=0.0, sigma=s)
                etas = np.array([-pu.eta_max, 0.0])
                Ra = rotation_profile(1.0, pu, ANOM, etas)
                R0 = rotation_profile(1.0, pu, 0.0, etas)
                tot.append(float(rotvec_from_matrix(Ra[-1])[1]))
                ano.append(float(rotvec_from_matrix(Ra[-1] @ R0[-1].T)[1]))
            e_t, o_t = split(tot)
            e_a, o_a = split(ano)
            inpulse.append(dict(a0=a0, sigma=s,
                                total_even_max=e_t, total_odd_max=o_t,
                                total_asym_break=e_t / max(o_t, 1e-300),
                                anom_even_max=e_a, anom_odd_max=o_a,
                                anom_asym_break=e_a / max(o_a, 1e-300),
                                total=tot, anom=ano, phi0=phis.tolist()))

    net_cir = []
    for a0 in cfg["t4_a0s"][1::2]:
        for s in cfg["t4_sigmas"]:
            v, th = [], []
            for p in phis:
                pu = Pulse(a0=a0, N=None, phi0=float(p), env="gauss",
                           delta=1.0, sigma=s)
                r = net_rotation(1.0, pu, rtol=1e-13, atol=1e-16)
                v.append(r["rz"])
                th.append(r["theta"])
            e, o = split(v)
            net_cir.append(dict(a0=a0, sigma=s, even_max=e, odd_max=o,
                                theta_max=float(np.max(th)),
                                theta_ptp=float(np.max(th) - np.min(th)),
                                rz=v, phi0=phis.tolist()))
    return dict(in_pulse=inpulse, net_circular=net_cir,
                wall_s=time.perf_counter() - t0)


def stage_t5(cfg):
    """Applicability window (ESTIMATES; no radiation is simulated).

      chi   = 2 (hbar w/mc^2) gamma a0                     head-on plane wave
      R_C   = (2/3) alpha a0 chi  per cycle  x N            classical RR
              [Di Piazza, Mueller, Hatsagortsyan, Keitel, RMP 84, 1177 (2012)]
      N_gam ~ alpha a0 N                                    photons emitted
      P_sf  ~ alpha a0 chi^2 N                              radiative spin flips
              [Seipt, Del Sorbo, Ridgers, Thomas, PRA 98, 023417 (2018);
               Li, Chen, Keitel et al., PRL 122, 154801 (2019)]
    """
    w = HBAR_OMEGA_EV / MC2_EV

    def lam_of(g):                       # lam = gamma(1+beta), head-on
        return g + math.sqrt(max(g ** 2 - 1.0, 0.0))

    def theta_cir(a0, N):                # measured law, delta = 1, gaussian
        return ANOM ** 2 * (a0 ** 2 / 2.0) * (SIG_PER_N * N) * math.sqrt(math.pi) / 2.0

    def line(gamma, a0, N):
        chi = w * lam_of(gamma) * a0
        return dict(gamma=gamma, a0=a0, N=N, lam=lam_of(gamma), chi=chi,
                    R_C_pulse=(2.0 / 3.0) * ALPHA * a0 * chi * N,
                    N_gamma=ALPHA * a0 * N,
                    P_radflip=ALPHA * a0 * chi ** 2 * N,
                    theta_circular=theta_cir(a0, N),
                    Pflip_circular=math.sin(theta_cir(a0, N) / 2.0) ** 2)

    rows = [line(g, a0, N) for g in cfg["gammas"] for a0 in cfg["a0s"]
            for N in cfg["Ns"]]
    cand = [line(g, a0, N)
            for g, a0, N in ((1.0, 10.0, 8.0), (1.0, 40.0, 8.0), (1.0, 75.0, 8.0),
                             (1.0, 40.0, 32.0), (1.0, 75.0, 32.0),
                             (10.0, 75.0, 8.0), (100.0, 75.0, 8.0),
                             (1e3, 10.0, 8.0), (1e3, 75.0, 8.0),
                             (1e3, 210.0, 10.0), (5e3, 10.0, 8.0))]
    return dict(hbar_omega_eV=HBAR_OMEGA_EV, chi_formula="chi = (hbar w/mc^2) lam a0",
                theta_law="Theta = anom^2 a0^2 sigma sqrt(pi)/4, sigma = 3.7745 N (delta=1)",
                rows=rows, candidates=cand, note="analytic estimates only")


# --------------------------------------------------------------------- t6
T6_CASES = (
    dict(a0=10.0, N=1.0, env="gauss", gamma=10.0),
    dict(a0=10.0, N=4.0, env="gauss", gamma=1.0),
    dict(a0=1.0, N=1.0, env="cos2", gamma=1.0),
)
# The carrier-envelope phase at which each row of TABLE S3 of the Supplemental
# Material was taken.  It is not the same for every row, which is why the stage
# records all three phases of stage t1 and reports the table row separately from
# the maximum over phases.
T6_TABLE_PHI0 = (math.pi / 2.0, 0.7, 0.7)
# SciPy's Runge-Kutta drivers refuse an rtol below 100 eps and silently raise it
# (a UserWarning), so the tightest column of TABLE S3 is not the tolerance that
# was asked for: rtol = 1e-14 is executed as 2.22e-14.
RTOL_FLOOR = 100.0 * float(np.finfo(float).eps)


def stage_t6(cfg):
    """Convergence of the plane-wave zero in the integrator tolerance.

    A null result is worth only as much as the evidence that the step size has
    stopped mattering.  The g = 2 computation is rerun at anom = 0 with rtol
    from 1e-9 to 1e-14 and atol = 1e-3 rtol, on the three configurations of
    TABLE S3, each at the three carrier-envelope phases of stage t1.  Reported
    per (case, tolerance): the value at each phi0, the maximum over phi0, the
    number of right-hand-side evaluations, and the invariants.

    The point of the stage is NOT that the number converges.  At a0 = 10 it does
    not: tightening the tolerance by two decades leaves |Theta_net| fluctuating
    inside a band while nfev triples, which is what a round-off floor looks like
    and is the evidence that the vanishing is a property of the equations.  The
    price is that the headline value 2.57e-13 rad is not stable to three digits;
    `spread_rel` below measures how unstable it is, and the defensible statement
    is the bound, not the digits.
    """
    t0 = time.perf_counter()
    rows = []
    for case in T6_CASES:
        for rtol in cfg["t6_rtols"]:
            atol = 1.0e-3 * rtol
            per = []
            for phi0 in cfg["t1_phis"]:
                pu = Pulse(a0=case["a0"], N=case["N"], phi0=phi0,
                           env=case["env"], delta=0.0)
                r = net_rotation(case["gamma"], pu, anom=0.0,
                                 rtol=rtol, atol=atol)
                per.append(dict(phi0=float(phi0), theta=float(r["theta"]),
                                nfev=int(r["nfev"]),
                                orth_err=float(r["orth_err"]),
                                du=float(r["du"])))
            rows.append(dict(case=dict(case), rtol=float(rtol), atol=float(atol),
                             rtol_effective=float(max(rtol, RTOL_FLOOR)),
                             per_phi0=per,
                             theta_max=max(p["theta"] for p in per),
                             theta_min=min(p["theta"] for p in per),
                             nfev_max=max(p["nfev"] for p in per),
                             orth_max=max(p["orth_err"] for p in per)))

    def series(i):
        return [r for r in rows if r["case"] == dict(T6_CASES[i])]

    def at_phi(r, phi0):
        j = min(range(len(r["per_phi0"])),
                key=lambda k: abs(r["per_phi0"][k]["phi0"] - phi0))
        return r["per_phi0"][j]

    summary = []
    for i in range(len(T6_CASES)):
        s = sorted(series(i), key=lambda r: -r["rtol"])
        phi_t = T6_TABLE_PHI0[i] if i < len(T6_TABLE_PHI0) else cfg["t1_phis"][0]
        tab = {f"{r['rtol']:.0e}": at_phi(r, phi_t)["theta"] for r in s}
        by = {r["rtol"]: at_phi(r, phi_t)["theta"] for r in s}
        work = by.get(1e-13)
        tail = [v for k, v in by.items() if k <= 1e-12]
        summary.append(dict(
            case=dict(T6_CASES[i]),
            table_phi0=float(phi_t),
            theta_by_rtol_at_table_phi0=tab,
            theta_max_by_rtol={f"{r['rtol']:.0e}": r["theta_max"] for r in s},
            nfev_by_rtol={f"{r['rtol']:.0e}": r["nfev_max"] for r in s},
            theta_at_working_rtol=work,
            tail_min=min(tail) if tail else None,
            tail_max=max(tail) if tail else None,
            spread_rel=(max(abs(v / work - 1.0) for v in tail)
                        if work and tail else None),
            monotone=(all(s[k]["theta_max"] >= s[k + 1]["theta_max"]
                          for k in range(len(s) - 1))),
            nfev_ratio=(s[-1]["nfev_max"] / s[0]["nfev_max"]
                        if s[0]["nfev_max"] else None)))
    return dict(rows=rows, summary=summary, phis=list(cfg["t1_phis"]),
                atol_rule="atol = 1e-3 * rtol", anom=0.0,
                rtol_floor=RTOL_FLOOR,
                note=("`theta_by_rtol_at_table_phi0` is TABLE S3 of the "
                      "Supplemental Material; `theta_max_by_rtol` is the "
                      "maximum over the three carrier-envelope phases"),
                wall_s=time.perf_counter() - t0)


# --------------------------------------------------------------------- t7
T7_WINDOW = 0.5          # rad; exponents are read only where Theta is small


def stage_t7(cfg):
    """Longitudinal spin projection: is Delta S_par really O(a_e^6)?

    Section III.E of the main text grades so(3)_{u0} by a Z_2 and concludes
    three things: the exponent along nhat carries only even powers of the
    anomaly, so the correction to the holonomy is O(a_e^4); the leading
    correction to the AXIS is the third-order in-plane term, so the axis is
    tilted out of nhat by psi = O(a_e); and the change in the projection of the
    spin on nhat is quadratic in that tilt, hence O(a_e^6).  Only the third
    statement is about an observable, and until now nothing in this deposit
    measured it.

    This stage measures it.  The electron starts at rest (gamma = 1, so the
    third leg of the triad IS nhat) with its spin ALONG nhat, i.e. in a helicity
    eigenstate, crosses a circularly polarised Gaussian pulse, and

        DS_par = R[2][2] - 1

    is read straight off the transported triad.  Because the exponent is 6 and
    the physical anomaly is 1.16e-3, the physical value is ~1e-20 and is not
    measurable in double precision; the anomaly is therefore inflated over more
    than a decade and the exponent is read from the scaling, exactly as the
    inflated-anomaly null tests of stage t1 are read.

    Also recorded, and deliberately NOT the headline: `dS_par_worst`, the
    largest change of the nhat projection over ALL initial spin directions.
    That one is O(a_e^3), not O(a_e^6), because a spin that starts in the
    polarisation plane feels the tilt at first order.  The helicity statement is
    about the longitudinal state and nothing else.

    Cross-check: for a rotation by Theta about an axis at an angle psi from
    nhat, DS_par = -(1 - cos Theta) sin^2 psi exactly.  The stage reports the
    relative agreement between the measured DS_par and that identity, which
    turns the exponent 6 into a statement about the two factors separately:
    Theta = O(a_e^2) and psi = O(a_e).
    """
    t0 = time.perf_counter()
    out = []
    for a0 in cfg["t7_a0s"]:
        rows = []
        for x in cfg["t7_x"]:
            A = x / a0            # the expansion parameter is a_e a0, not a_e
            pu = Pulse(a0=a0, N=cfg["t7_N"], phi0=0.0, env="gauss", delta=1.0)
            r = net_rotation(1.0, pu, anom=A, rtol=1e-13, atol=1e-16,
                             with_matrix=True)
            R = np.array(r["R"])
            rv = np.array([r["rx"], r["ry"], r["rz"]])
            th = float(np.linalg.norm(rv))
            psi = float(math.atan2(math.hypot(rv[0], rv[1]), abs(rv[2])))
            dS = float(R[2, 2] - 1.0)
            worst = float(np.linalg.norm(R[2, :] - np.array([0.0, 0.0, 1.0])))
            ident = -(1.0 - math.cos(th)) * math.sin(psi) ** 2
            rows.append(dict(anom=float(A), x=float(x), a0=float(a0),
                             N=cfg["t7_N"],
                             theta=th, psi=psi, dS_par=dS, dS_par_worst=worst,
                             dS_par_identity=ident,
                             identity_rel_err=(abs(dS / ident - 1.0)
                                               if ident != 0.0 else None),
                             area=pu.swept_area(),
                             orth_err=float(r["orth_err"]),
                             nfev=int(r["nfev"])))

        # The perturbative ordering is only meaningful while the net angle is
        # small: at Theta ~ pi the angle read from an SO(3) matrix wraps (see
        # holonomy_scan.py, stage `wrap`) and every exponent read off it is
        # meaningless.  Exponents are therefore reported over the window
        # Theta <= T7_WINDOW, and the full grid is kept in `rows`.
        win = [r for r in rows if r["theta"] <= T7_WINDOW]

        def loglog(key, sub):
            xx = np.array([r["anom"] for r in sub])
            y = np.abs(np.array([r[key] for r in sub]))
            return dict(
                lowest_pair=float(math.log(y[1] / y[0])
                                  / math.log(xx[1] / xx[0])),
                endpoints=float(math.log(y[-1] / y[0])
                                / math.log(xx[-1] / xx[0])),
                local=[[float(xx[i]), float(xx[i + 1]),
                        float(math.log(y[i + 1] / y[i])
                              / math.log(xx[i + 1] / xx[i]))]
                       for i in range(len(xx) - 1)])

        # extrapolation to the physical anomaly from the smallest inflated point
        r0 = rows[0]
        coeff = abs(r0["dS_par"]) / r0["anom"] ** 6
        out.append(dict(a0=float(a0), rows=rows,
                        n_in_window=len(win), theta_window=T7_WINDOW,
                        scaling={k: loglog(k, win) for k in
                                 ("dS_par", "theta", "psi", "dS_par_worst")},
                        scaling_full={k: loglog(k, rows) for k in
                                      ("dS_par", "theta")},
                        max_identity_rel_err=max(
                            r["identity_rel_err"] for r in win
                            if r["identity_rel_err"] is not None),
                        floor_orth_err=max(r["orth_err"] for r in rows),
                        coeff_a6=float(coeff),
                        coeff_over_a0_6=float(coeff / a0 ** 6),
                        dS_par_at_physical_anomaly=float(-coeff * ANOM ** 6)))
    return dict(series=out, anomaly_physical=ANOM,
                observable=("DS_par = R[2][2] - 1, initial spin along nhat, "
                            "gamma = 1, circular polarisation, Gaussian "
                            "envelope; the scan is at fixed a_e*a0 across a0"),
                wall_s=time.perf_counter() - t0)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="")
    ap.add_argument("--merge", default="")
    ap.add_argument("--out", default="spin_magnitude.json")
    ap.add_argument("--n-phi-lin", type=int, default=4)
    ap.add_argument("--n-phi-cir", type=int, default=8)
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-14)
    ap.add_argument("--gammas", default="1,10,100,1000,5000")
    ap.add_argument("--a0s", default="0.1,1,10")
    ap.add_argument("--Ns", default="1,2,4,8")
    ap.add_argument("--deltas", default="0,1")
    # Per-stage grids.  EVERY default below reproduces the published run
    # exactly; they exist so that `reproduce_all.py --quick` can shrink t1, t4,
    # t6 and t7 instead of running them at full size, which is what it used to
    # do for t1 (83.9 s in "quick" mode, i.e. the full grid).
    ap.add_argument("--t1-envs", default="gauss,cos2")
    ap.add_argument("--t1-gammas", default="1,10,100,1000,5000")
    ap.add_argument("--t1-a0s", default="0.1,1,10")
    ap.add_argument("--t1-Ns", default="1,4")
    ap.add_argument("--t1-phis", default="0,0.7,1.5707963267948966")
    ap.add_argument("--t1-anoms",
                    default="1.15965218059e-3,1e-2,1e-1,1,10")
    ap.add_argument("--t1-cf-gammas", default="1,100,1000")
    # stage t3(f) only
    ap.add_argument("--t3-sigmas",
                    default="0.25,0.5,0.75,1,1.5,2,2.5,3,3.7745,5,7.549")
    ap.add_argument("--t3-nphi", type=int, default=32)
    ap.add_argument("--t4-a0s", default="0.01,0.1,0.3,1,3,10")
    ap.add_argument("--t4-sigmas", default="1,3.7745")
    ap.add_argument("--t4-nphi", type=int, default=16)
    ap.add_argument("--t6-rtols", default="1e-9,1e-10,1e-11,1e-12,1e-13,1e-14")
    ap.add_argument("--t7-x", default="0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.5",
                    help="grid in the expansion parameter a_e*a0 (stage t7)")
    ap.add_argument("--t7-a0s", default="1,3,10")
    ap.add_argument("--t7-N", type=float, default=1.0)
    args = ap.parse_args(argv)

    np.random.seed(SEED)

    def flist(s):
        return [float(x) for x in s.split(",") if x.strip()]

    cfg = dict(n_phi_lin=args.n_phi_lin, n_phi_cir=args.n_phi_cir,
               rtol=args.rtol, atol=args.atol,
               gammas=flist(args.gammas), a0s=flist(args.a0s),
               Ns=flist(args.Ns), deltas=flist(args.deltas),
               t1_envs=[x.strip() for x in args.t1_envs.split(",") if x.strip()],
               t1_gammas=flist(args.t1_gammas), t1_a0s=flist(args.t1_a0s),
               t1_Ns=flist(args.t1_Ns), t1_phis=flist(args.t1_phis),
               t1_anoms=flist(args.t1_anoms),
               t1_cf_gammas=flist(args.t1_cf_gammas),
               t3_sigmas=flist(args.t3_sigmas), t3_nphi=args.t3_nphi,
               t4_a0s=flist(args.t4_a0s), t4_sigmas=flist(args.t4_sigmas),
               t4_nphi=args.t4_nphi,
               t6_rtols=flist(args.t6_rtols),
               t7_x=flist(args.t7_x), t7_a0s=flist(args.t7_a0s),
               t7_N=args.t7_N)

    if args.merge:
        res = {}
        for fn in args.merge.split(","):
            with open(fn.strip(), encoding="utf-8") as fh:
                res.update(json.load(fh))
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print("merged ->", args.out)
        return res

    res = dict(meta=dict(script="spin_magnitude.py", anomaly=ANOM, seed=SEED,
                         cfg=cfg, python=sys.version.split()[0],
                         numpy=np.__version__, scipy=scipy.__version__,
                         os=platform.platform(), sigma_per_N=SIG_PER_N,
                         L_per_N=L_PER_N, n_sigma_trunc=N_SIGMA_TRUNC))
    fns = dict(t1=stage_t1, t2=stage_t2, t3=stage_t3, t4=stage_t4, t5=stage_t5,
               t6=stage_t6, t7=stage_t7)
    for s in args.stages.split(","):
        s = s.strip()
        if not s:
            continue
        t = time.perf_counter()
        res[s] = fns[s](cfg)
        print(f"[{s}] done in {time.perf_counter() - t:.1f} s", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("written", args.out, flush=True)
    return res


if __name__ == "__main__":
    main()
