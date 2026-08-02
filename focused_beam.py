# -*- coding: utf-8 -*-
r"""
focused_beam.py -- does a finite waist break the plane-wave spin holonomy theorem,
and by how much?

Companion to `spin_magnitude.py` (exact plane wave, Volkov orbit, algebraic) and to
plan/14-theorem-verification.md.  There the net spin rotation after a finite pulse
was shown to be a *holonomy*,

    Theta_net = -(1/2) a^2 A,   A = int (a_x a_y' - a_y a_x') d eta,  a = (g-2)/2,

which vanishes identically for linear polarisation and is exactly CEP-independent
for circular polarisation.  The proof uses A^mu = A^mu(eta) ONLY.  A finite waist
w0 destroys that hypothesis.  This script measures what survives.

================================================================================
1. FIELD MODEL -- Lax-Louisell-McKnight expansion in eps = 1/(k w0)
================================================================================
Units: k = omega = c = 1, so lengths are k x and times are omega t.  The diffraction
parameter is

    eps = 1/(k w0) = lambda/(2 pi w0),      z_R = k w0^2/2 = 1/(2 eps^2).

The beam is built from a Coulomb-gauge vector potential, normalised as in
spin_magnitude.py:  a^mu = q A^mu/(m c^2), so that a0 = |q|E0/(m c omega).

Transverse complex amplitude (LLM / Davis expansion; the standard strong-field form
is Y. I. Salamin, "Fields of a Gaussian beam beyond the paraxial approximation",
Appl. Phys. B 86, 319 (2007), and Y. I. Salamin & C. H. Keitel, Phys. Rev. Lett.
88, 095005 (2002); the same hierarchy is Lax, Louisell & McKnight, Phys. Rev. A 11,
1365 (1975)):

    s = eps^2 (X^2+Y^2) = rho^2/w0^2 ,   zeta = 2 eps^2 Z = z/z_R ,  f = 1/(1+i zeta)
    psi_0 = f exp(-f s)                                   (paraxial mode)
    psi   = psi_0 + eps^2 chi ,
    chi   = exp(-f s) [ c0 f^2 + (2-c0) f^3 s - f^4 s^2 ]                       (*)

(*) is derived here, not copied: substituting psi_0 + eps^2 chi into the wave
equation and demanding that the O(eps^2) residual cancel gives, in the variables
(s, zeta),

    s chi_ss + chi_s + i chi_zeta = -psi_0,zetazeta = exp(-f s)(f^5 s^2 - 4 f^4 s + 2 f^3),

whose general solution is (*) with c0 free -- the free constant multiplies
d psi_0/d zeta, a homogeneous (paraxial) solution, i.e. an O(eps^2) refocusing
ambiguity intrinsic to the expansion.  Default convention c0 = 0, i.e. the
correction vanishes on the axis, so a0 keeps its meaning "peak normalised potential
at the focus".  `--c0 1` re-runs with a different member of the family; the spread
is the systematic error of the field model at this order (stage `field`).

Pulse and polarisation.  With eta = t - Z, envelope g(eta) = exp(-eta^2/2 sigma^2),
sigma = pi N/sqrt(ln 2) (N = cycles in the FWHM of the intensity envelope, same
convention as spin_magnitude.py), carrier C = exp(-i(eta+phi0)), and
alpha = (1, i delta)/sqrt(1+delta^2)  (delta = 0 linear, delta = 1 circular),

    a_x^c = a0 alpha_x psi g C ,      a_y^c = a0 alpha_y psi g C ,
    a_z^c = a0 W C ,   W = i (D g + dW/dZ) iterated,   D = alpha_x psi_,X + alpha_y psi_,Y

and a^mu = (0, Re a_x^c, Re a_y^c, Re a_z^c).  The a_z closure solves the
Coulomb-gauge condition div a = 0 by fixed-point iteration starting from
W_0 = i D g: the leading term is O(eps), each further iteration adds O(eps^3) and
O(eps/sigma) pieces, so with GAUGE_ITER = 2 the longitudinal field is complete
through eps^3 (and through eps/sigma^2) as required.  In the limit eps -> 0 this reduces
*exactly* to the plane-wave pulse of spin_magnitude.py:
a_x -> a0 g cos(eta+phi0)/sqrt(1+delta^2), a_y -> a0 delta g sin(eta+phi0)/sqrt(1+delta^2).

Fields: E = -da/dt, B = curl a  (scalar potential zero).  Therefore
    div B = 0                      exactly (machine precision),
    curl E + dB/dt = 0             exactly (machine precision),
    div E = -d(div a)/dt           residual, measured,
    curl B - dE/dt = -box a        residual, measured.
Stage `field` measures both residuals as a function of eps and of the truncation
order and confirms the expected scaling.

================================================================================
2. DYNAMICS -- Lorentz + Thomas-BMT in LAB TIME, no wave approximations
================================================================================
State y = (x, u, {S^mu}) with u^i the spatial part of the 4-velocity
(u^0 = sqrt(1+u.u), so u.u = 1 is exact by construction, not by integration).

    dx/dt = beta = u/gamma
    du/dt = E + beta x B
    dS^mu/dt = (1/gamma) [ (1+a) (f S)^mu + a u^mu (S . f u) ] ,
    (f v)^0 = E . v_vec ,   (f v)^i = v^0 E^i + (v_vec x B)^i

which is Jackson (11.170) divided by gamma (d/dtau = gamma d/dt), with a = (g-2)/2
and the fields already normalised by q/(m c omega).  Several anomalies are
propagated simultaneously through the same orbit (the spin does not react back:
the Stern-Gerlach force is smaller by hbar k/mc).

Ground truth: scipy DOP853.  Stage `conv` shows tolerance convergence and the
conservation of u.u, S.u, S.S along the trajectory.

================================================================================
3. OBSERVABLE -- spin rotation RELATIVE TO THE MOMENTUM
================================================================================
Unlike the plane wave, here u(+inf) != u(-inf): the electron is scattered
ponderomotively, and the associated Wigner rotation is a large, purely kinematic
effect which must be removed (plan/14 Sec. 4.1).  We propagate the three rest-frame
basis 4-vectors e_i(t) (each obeying T-BMT), and compare the final triad with the
triad obtained by *pure-boosting* the initial one onto the final momentum:

    e~_i = B(u_0 -> u_f) e_i(t_i),     R_ij = -<e~_i, e_j(t_f)>_Minkowski
    r    = rotvec(R)          (rotation of the spin frame w.r.t. the momentum)

In the plane-wave limit u_f = u_0, B = 1, and r reduces exactly to the plane-wave
net rotation of spin_magnitude.py.

Anomaly expansion.  The same orbit is used for a grid of anomalies
a in {0, +h, -h, +2h, -2h}, and r(a) is expanded by 5-point finite differences:

    r(a) = r0 + a r1 + a^2 r2 + O(a^3)

  * r0 = r(0)  -- the g = 2 momentum-relative rotation.  Zero in a plane wave
    (exactly), possibly non-zero under focusing.
  * r1        -- FIRST order in the anomaly.  Identically zero in a plane wave
    (plan/14 Sec. 1.5).  Non-zero r1 is the signature that focusing broke the
    theorem, and it is the quantity expected to carry CEP dependence.
  * r2        -- must converge to -(A/2) z^ as eps -> 0 (the holonomy).

Stages
    field    Maxwell residuals of the field model vs eps and truncation order
    conv     tolerance convergence + conservation of u.u, S.u, S.S
    order    systematics of the field model (order 1 vs 3, c0 = 0 vs 1)
    eps      MAIN scan: r0, r1, r2 vs the diffraction parameter
    xcheck   direct run at the physical anomaly vs the plane-wave holonomy
    cep      16-point CEP scan (harmonic content)
    cepscan  CEP-odd amplitude vs eps, a0, N, gamma, impact parameter
    a0big    CEP amplitude up to a0 = 24 (linear polarisation)
    scans    a0, gamma, N, impact parameter, ellipticity
    window   applicability window (ANALYTIC ESTIMATES, nothing simulated)

Usage (this is exactly how focused_beam.json/.log were produced)
    python focused_beam.py --stages field,conv,order,eps,xcheck,scans,window --out fb_A.json
    python focused_beam.py --stages cep,cepscan                              --out fb_B.json
    python focused_beam.py --stages a0big                                    --out fb_C.json
    python focused_beam.py --merge fb_A.json,fb_B.json,fb_C.json --out focused_beam.json
    python focused_beam_summary.py focused_beam.json          # the tables in the .log
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
import sympy as sp
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------- constants
ANOM = 1.15965218059e-3          # a = (g-2)/2, CODATA 2022 electron
ALPHA = 7.2973525693e-3
MC2_EV = 510998.95
HBAR_OMEGA_EV = 1.55             # Ti:sapphire, 800 nm
SEED = 20260730

SIG_PER_N = math.pi / math.sqrt(math.log(2.0))            # sigma = 3.7745 N
GAUGE_ITER = 2                                            # Coulomb-gauge iterations
NSIG = 7.0                                                # truncation |eta| < NSIG sigma

ETA4 = np.diag([1.0, -1.0, -1.0, -1.0])


# ================================================================ field
_FIELD_CACHE = {}


def build_field(order=3, want_residuals=False):
    """Symbolically build (E, B) and, optionally, the four Maxwell residuals.

    Returns lambdified callables of (t, X, Y, Z, eps, a0, delta, phi0, sigma, c0).
    order = 1 : psi = psi_0                 (transverse O(eps^0), E_z O(eps))
    order = 3 : psi = psi_0 + eps^2 chi     (transverse O(eps^2), E_z O(eps^3))
    """
    key = (order, want_residuals)
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]
    t, X, Y, Z = sp.symbols("t X Y Z", real=True)
    eps, a0, dl, ph, sg, c0 = sp.symbols("eps a0 delta phi0 sigma c0", real=True)
    eta = t - Z
    C = sp.exp(-sp.I * (eta + ph))
    s = eps ** 2 * (X ** 2 + Y ** 2)
    ze = 2 * eps ** 2 * Z
    f = 1 / (1 + sp.I * ze)
    psi0 = f * sp.exp(-f * s)
    chi = sp.exp(-f * s) * (c0 * f ** 2 + (2 - c0) * f ** 3 * s - f ** 4 * s ** 2)
    psi = psi0 + eps ** 2 * chi if order >= 3 else psi0
    nrm = 1 / sp.sqrt(1 + dl ** 2)
    ax_, ay_ = nrm, sp.I * dl * nrm
    g = sp.exp(-eta ** 2 / (2 * sg ** 2))
    D = ax_ * sp.diff(psi, X) + ay_ * sp.diff(psi, Y)
    # Coulomb-gauge closure  div a = 0  ->  W = i (D g + dW/dZ), iterated.
    # W_0 = i D g is the O(eps) longitudinal amplitude; each further iteration adds
    # O(eps^2) and O(1/sigma) corrections.  GAUGE_ITER = 2 keeps everything through
    # eps^3 and through eps/sigma^2.
    W = sp.I * D * g
    for _ in range(GAUGE_ITER):
        W = sp.I * (D * g + sp.diff(W, Z))
    Ac = [a0 * ax_ * psi * g * C,
          a0 * ay_ * psi * g * C,
          a0 * W * C]
    Ec = [-sp.diff(Ac[i], t) for i in range(3)]
    Bc = [sp.diff(Ac[2], Y) - sp.diff(Ac[1], Z),
          sp.diff(Ac[0], Z) - sp.diff(Ac[2], X),
          sp.diff(Ac[1], X) - sp.diff(Ac[0], Y)]
    args = (t, X, Y, Z, eps, a0, dl, ph, sg, c0)
    fEB = sp.lambdify(args, Ec + Bc, modules="numpy", cse=True)
    fres = None
    if want_residuals:
        V = (X, Y, Z)
        divE = sum(sp.diff(Ec[i], V[i]) for i in range(3))
        divB = sum(sp.diff(Bc[i], V[i]) for i in range(3))
        curlB = [sp.diff(Bc[2], Y) - sp.diff(Bc[1], Z),
                 sp.diff(Bc[0], Z) - sp.diff(Bc[2], X),
                 sp.diff(Bc[1], X) - sp.diff(Bc[0], Y)]
        curlE = [sp.diff(Ec[2], Y) - sp.diff(Ec[1], Z),
                 sp.diff(Ec[0], Z) - sp.diff(Ec[2], X),
                 sp.diff(Ec[1], X) - sp.diff(Ec[0], Y)]
        amp = [curlB[i] - sp.diff(Ec[i], t) for i in range(3)]
        far = [curlE[i] + sp.diff(Bc[i], t) for i in range(3)]
        fres = sp.lambdify(args, [divE, divB] + amp + far, modules="numpy", cse=True)
    _FIELD_CACHE[key] = (fEB, fres)
    return _FIELD_CACHE[key]


class Beam:
    """Parameter bundle + fast real-valued field evaluation."""

    def __init__(self, eps=0.1, a0=1.0, delta=1.0, phi0=0.0, N=1.0, order=3, c0=0.0,
                 nsig=NSIG):
        self.eps, self.a0, self.delta, self.phi0 = float(eps), float(a0), float(delta), float(phi0)
        self.N = float(N)
        self.sigma = SIG_PER_N * self.N
        self.order, self.c0, self.nsig = int(order), float(c0), float(nsig)
        self.eta_max = self.nsig * self.sigma
        self.w0 = 1.0 / self.eps
        self.zR = 0.5 / self.eps ** 2
        self._fEB = build_field(self.order)[0]
        self._p = (self.eps, self.a0, self.delta, self.phi0, self.sigma, self.c0)

    def EB(self, t, x):
        v = self._fEB(t, x[0], x[1], x[2], *self._p)
        return np.real(np.asarray(v, dtype=complex))

    def swept_area(self):
        """Plane-wave holonomy area A = int (a_x a_y' - a_y a_x') deta, truncated."""
        d = self.delta
        s = self.sigma
        return (d / (1.0 + d ** 2)) * self.a0 ** 2 * s * math.sqrt(math.pi) \
            * math.erf(self.nsig)

    def as_dict(self):
        return dict(eps=self.eps, a0=self.a0, delta=self.delta, phi0=self.phi0,
                    N=self.N, sigma=self.sigma, order=self.order, c0=self.c0,
                    nsig=self.nsig, w0_over_lam=self.w0 / (2 * math.pi),
                    zR=self.zR, area=self.swept_area())


# ================================================================ kinematics
def mink(v, w):
    return v[0] * w[0] - v[1] * w[1] - v[2] * w[2] - v[3] * w[3]


def triad_of(u3):
    """Three orthonormal rest-frame 4-vectors for spatial 4-velocity part u3."""
    g = math.sqrt(1.0 + float(u3 @ u3))
    out = []
    for n in np.eye(3):
        un = float(u3 @ n)
        out.append(np.concatenate(([un], n + (un / (g + 1.0)) * u3)))
    return np.array(out)


def boost_matrix(u0, u1):
    """Pure boost B with B u0 = u1 (4-vectors, mixed index)."""
    gg = 1.0 + mink(u1, u0)
    s = u1 + u0
    return np.eye(4) - np.outer(s, s @ ETA4) / gg + 2.0 * np.outer(u1, u0 @ ETA4)


def rotvec(R):
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = float(np.linalg.norm(w))
    c = 0.5 * (float(np.trace(R)) - 1.0)
    th = math.atan2(s, c)
    if s < 1e-300:
        return w
    return w * (th / s)


# ================================================================ trajectory
def trajectory(beam, gamma=1.0, anoms=(0.0,), b=0.0, by=0.0, rtol=1e-11, atol=1e-13,
               n_track=0):
    """Integrate Lorentz + T-BMT in lab time; return net rotation data.

    Head-on geometry: wave along +z, electron along -z with Lorentz factor gamma.
    The electron is launched so that it reaches z = 0 (the focus) at t = 0, which is
    also the peak of the pulse; the impact parameter is (b, by).
    """
    p = math.sqrt(max(gamma ** 2 - 1.0, 0.0))
    beta = p / gamma
    ti = -beam.eta_max / (1.0 + beta)
    Zi = -beta * ti
    u0 = np.array([0.0, 0.0, -p])
    E0 = triad_of(u0)
    na = len(anoms)
    An = np.asarray(anoms, float)
    y0 = np.concatenate(([b, by, Zi], u0, np.tile(E0.ravel(), na)))
    fEB, pars = beam._fEB, beam._p

    def rhs(t, y):
        u = y[3:6]
        gg = math.sqrt(1.0 + u @ u)
        be = u / gg
        v = np.real(np.asarray(fEB(t, y[0], y[1], y[2], *pars), dtype=complex))
        Ef, Bf = v[0:3], v[3:6]
        du = Ef + np.cross(be, Bf)
        S = y[6:].reshape(na, 3, 4)
        S0, Sv = S[..., 0], S[..., 1:]
        w0 = float(Ef @ u)
        wv = gg * Ef + np.cross(u, Bf)
        sc = S0 * w0 - Sv @ wv                       # S . (f u)
        oS0 = Sv @ Ef                                # (f S)^0
        oSv = S0[..., None] * Ef + np.cross(Sv, Bf)  # (f S)^i
        c1 = (1.0 + An)[:, None]
        dS = np.empty_like(S)
        dS[..., 0] = (c1 * oS0 + An[:, None] * sc * gg) / gg
        dS[..., 1:] = (c1[..., None] * oSv
                       + (An[:, None] * sc)[..., None] * u) / gg
        return np.concatenate((be, du, dS.ravel()))

    te = np.linspace(ti, -ti, n_track) if n_track else None
    sol = solve_ivp(rhs, (ti, -ti), y0, method="DOP853", rtol=rtol, atol=atol,
                    t_eval=te)
    if not sol.success:
        raise RuntimeError(sol.message)
    yf = sol.y[:, -1]
    uf3 = yf[3:6]
    gf = math.sqrt(1.0 + uf3 @ uf3)
    U0 = np.concatenate(([gamma], u0))
    Uf = np.concatenate(([gf], uf3))
    Bm = boost_matrix(U0, Uf)
    Eref = np.array([Bm @ E0[j] for j in range(3)])
    Sf = yf[6:].reshape(na, 3, 4)
    out = dict(anoms=list(map(float, An)), gamma=gamma, b=b, by=by,
               nfev=int(sol.nfev), nsteps=int(sol.t.size))
    rs, orth, su, ss = [], [], [], []
    for m in range(na):
        R = np.array([[-mink(Eref[j], Sf[m][i]) for i in range(3)] for j in range(3)])
        rs.append(rotvec(R).tolist())
        orth.append(float(np.max(np.abs(R.T @ R - np.eye(3)))))
        su.append(float(np.max(np.abs([mink(Sf[m][i], Uf) for i in range(3)]))))
        ss.append(float(np.max(np.abs([mink(Sf[m][i], Sf[m][i]) + 1.0
                                       for i in range(3)]))))
    out["r"] = rs
    out["orth_err"] = float(max(orth))
    out["su_err"] = float(max(su))
    out["ss_err"] = float(max(ss))
    out["du"] = [float(x) for x in (uf3 - u0)]
    out["dgamma"] = float(gf - gamma)
    out["du_perp"] = float(math.hypot(uf3[0], uf3[1]))
    out["defl_angle"] = float(math.atan2(math.hypot(uf3[0], uf3[1]), abs(uf3[2]))) \
        if gamma > 1 else float("nan")
    out["xf"] = [float(x) for x in yf[0:3]]
    E_i = beam.EB(ti, np.array([b, by, Zi]))
    E_f = beam.EB(-ti, yf[0:3])
    out["field_end"] = float(max(np.max(np.abs(E_i)), np.max(np.abs(E_f))))
    if n_track:
        out["t_track"] = sol.t.tolist()
        out["y_track"] = sol.y
    return out


def anomaly_expansion(beam, gamma=1.0, h=0.05, b=0.0, by=0.0, rtol=1e-11, atol=1e-13):
    """r(a) = r0 + a r1 + a^2 r2 + ... by 5-point finite differences in a."""
    anoms = (0.0, h, -h, 2 * h, -2 * h)
    tr = trajectory(beam, gamma=gamma, anoms=anoms, b=b, by=by, rtol=rtol, atol=atol)
    r = np.array(tr["r"])
    r0, rp, rm, rp2, rm2 = r
    r1 = (8 * (rp - rm) - (rp2 - rm2)) / (12 * h)
    r2 = 0.5 * (-rm2 + 16 * rm - 30 * r0 + 16 * rp - rp2) / (12 * h ** 2)
    tr["r0"] = r0.tolist()
    tr["r1"] = r1.tolist()
    tr["r2"] = r2.tolist()
    tr["r0_abs"] = float(np.linalg.norm(r0))
    tr["r1_abs"] = float(np.linalg.norm(r1))
    tr["r2_abs"] = float(np.linalg.norm(r2))
    tr["h"] = h
    tr["area"] = beam.swept_area()
    tr["r2_pw_z"] = -0.5 * beam.swept_area()
    tr.pop("r")
    tr["r_all"] = r.tolist()
    return tr


def powerlaw(xs, ys):
    """Local log-log slopes and a global least-squares slope (positive data only)."""
    x = np.asarray(xs, float)
    y = np.abs(np.asarray(ys, float))
    m = (x > 0) & (y > 0) & np.isfinite(y)
    if m.sum() < 2:
        return dict(slope=None, local=[])
    lx, ly = np.log(x[m]), np.log(y[m])
    A = np.vstack([lx, np.ones_like(lx)]).T
    sl, c = np.linalg.lstsq(A, ly, rcond=None)[0]
    loc = [(float(x[m][i]), float(x[m][i + 1]),
            float((ly[i + 1] - ly[i]) / (lx[i + 1] - lx[i])))
           for i in range(m.sum() - 1)]
    return dict(slope=float(sl), amp=float(math.exp(c)), local=loc)


# ================================================================ stages
def stage_field(cfg):
    """Maxwell test: residuals of the four equations vs eps and truncation order.

    The sample points are held fixed in the SIMILARITY variables
    (rho/w0, z/z_R, eta/sigma), so that comparing different eps compares the same
    physical location inside the beam; residuals are normalised by a0 (= the peak
    field), which is the natural dimensionless measure because [residual] = k [E].
    Two pulse lengths are used: the physical one (sigma = 3.77, N = 1) and a
    quasi-monochromatic one (sigma = 100).  The envelope contributes an extra
    O(eps^2/sigma) term to the Ampere residual which has nothing to do with the
    LLM hierarchy; the long pulse isolates the hierarchy itself.
    """
    t0 = time.perf_counter()
    res = {}
    grid = [(rr, zz, ee)
            for rr in (0.0, 0.5, 1.0, 1.5)
            for zz in (0.0, 0.5, 1.0, 2.0)
            for ee in (0.0, 0.7, 1.5)]
    for sg_tag, sg in (("pulse", SIG_PER_N * 1.0), ("long", 100.0)):
        for order in (1, 3):
            fEB, fres = build_field(order, want_residuals=True)
            rows = []
            for eps in cfg["field_eps"]:
                acc = dict(divE=0.0, divB=0.0, amp=0.0, far=0.0, Emax=0.0, Ez=0.0)
                for (rr, zz, ee) in grid:
                    X = rr / eps / math.sqrt(2.0)
                    Yv = X
                    Zv = zz * 0.5 / eps ** 2
                    t = ee * sg + Zv
                    a = (t, X, Yv, Zv, eps, 1.0, 1.0, 0.0, sg, 0.0)
                    EB = np.real(np.asarray(fEB(*a), dtype=complex))
                    R = np.real(np.asarray(fres(*a), dtype=complex))
                    acc["Emax"] = max(acc["Emax"], float(np.max(np.abs(EB[0:3]))))
                    acc["Ez"] = max(acc["Ez"], abs(float(EB[2])))
                    acc["divE"] = max(acc["divE"], abs(float(R[0])))
                    acc["divB"] = max(acc["divB"], abs(float(R[1])))
                    acc["amp"] = max(acc["amp"], float(np.max(np.abs(R[2:5]))))
                    acc["far"] = max(acc["far"], float(np.max(np.abs(R[5:8]))))
                rows.append(dict(eps=eps, order=order, sigma=sg, **acc,
                                 divE_rel=acc["divE"], amp_rel=acc["amp"],
                                 Ez_rel=acc["Ez"] / acc["Emax"]))
            tag = f"{sg_tag}_order{order}"
            res[tag] = rows
            res[tag + "_scaling"] = dict(
                divE=powerlaw([r["eps"] for r in rows], [r["divE_rel"] for r in rows]),
                ampere=powerlaw([r["eps"] for r in rows], [r["amp_rel"] for r in rows]),
                Ez=powerlaw([r["eps"] for r in rows], [r["Ez_rel"] for r in rows]))
    # plane-wave limit check: at eps -> 0 the field must equal the exact plane wave
    pw = []
    for eps in (1e-2, 1e-3, 1e-4):
        bm = Beam(eps=eps, a0=1.0, delta=1.0, phi0=0.3, N=1.0)
        err = 0.0
        for et in (-3.0, -1.0, 0.0, 1.0, 3.0):
            E = bm.EB(et, np.zeros(3))
            g = math.exp(-et ** 2 / (2 * bm.sigma ** 2))
            ph = et + 0.3
            n = 1.0 / math.sqrt(2.0)
            # plane wave: a = a0 n (g cos ph, delta g sin ph, 0); E = -da/dt
            dg = -et / bm.sigma ** 2 * g
            Ex = -(dg * math.cos(ph) - g * math.sin(ph)) * n
            Ey = -(dg * math.sin(ph) + g * math.cos(ph)) * n
            err = max(err, abs(E[0] - Ex), abs(E[1] - Ey), abs(E[2]))
        pw.append(dict(eps=eps, max_abs_err=float(err)))
    res["plane_wave_limit"] = pw
    res["wall_s"] = time.perf_counter() - t0
    return res


def stage_conv(cfg):
    """Tolerance convergence and conservation of u.u, S.u, S.S along the orbit."""
    t0 = time.perf_counter()
    rows = []
    for (eps, a0, gam, dl) in cfg["conv_cases"]:
        bm = Beam(eps=eps, a0=a0, delta=dl, phi0=0.3, N=1.0)
        ref = None
        sub = []
        for rtol, atol in ((1e-8, 1e-10), (1e-10, 1e-12), (1e-11, 1e-13),
                           (1e-12, 1e-14), (1e-13, 1e-15)):
            r = anomaly_expansion(bm, gamma=gam, h=cfg["h"], rtol=rtol, atol=atol)
            v = np.array(r["r1"])
            if ref is None:
                pass
            sub.append(dict(rtol=rtol, atol=atol, r0=r["r0"], r1=r["r1"], r2=r["r2"],
                            nfev=r["nfev"], su=r["su_err"], ss=r["ss_err"],
                            orth=r["orth_err"], du_perp=r["du_perp"]))
        ref = np.array(sub[-1]["r1"])
        for s in sub:
            d = np.array(s["r1"]) - ref
            s["r1_err_vs_tightest"] = float(np.linalg.norm(d))
            s["r1_rel_err"] = float(np.linalg.norm(d) / max(np.linalg.norm(ref), 1e-300))
            d0 = np.array(s["r0"]) - np.array(sub[-1]["r0"])
            s["r0_err_vs_tightest"] = float(np.linalg.norm(d0))
        rows.append(dict(eps=eps, a0=a0, gamma=gam, delta=dl, tol_scan=sub))
    # invariants along a trajectory
    bm = Beam(eps=0.15, a0=2.0, delta=1.0, phi0=0.3, N=1.0)
    tr = trajectory(bm, gamma=10.0, anoms=(0.0, 0.1), rtol=1e-12, atol=1e-14,
                    n_track=401)
    Ytr = tr.pop("y_track")
    uu, su, ss = [], [], []
    for m in range(Ytr.shape[1]):
        u3 = Ytr[3:6, m]
        g = math.sqrt(1.0 + u3 @ u3)
        U = np.concatenate(([g], u3))
        uu.append(abs(mink(U, U) - 1.0))
        S = Ytr[6:, m].reshape(2, 3, 4)
        for q in range(2):
            for i in range(3):
                su.append(abs(mink(S[q][i], U)))
                ss.append(abs(mink(S[q][i], S[q][i]) + 1.0))
    inv = dict(max_uu=float(max(uu)), max_Su=float(max(su)), max_SS=float(max(ss)),
               case=dict(eps=0.15, a0=2.0, gamma=10.0, delta=1.0, rtol=1e-12))
    return dict(rows=rows, invariants=inv, wall_s=time.perf_counter() - t0)


def stage_eps(cfg):
    """MAIN SCAN: anomaly-resolved rotation vs the diffraction parameter eps."""
    t0 = time.perf_counter()
    res = {}
    for tag, (gam, a0, dl) in cfg["eps_cases"].items():
        rows = []
        for eps in cfg["eps_list"]:
            bm = Beam(eps=eps, a0=a0, delta=dl, phi0=cfg["phi0"], N=cfg["N"])
            r = anomaly_expansion(bm, gamma=gam, h=cfg["h"], rtol=cfg["rtol"],
                                  atol=cfg["atol"])
            r["beam"] = bm.as_dict()
            r["r2_dev"] = float(np.linalg.norm(
                np.array(r["r2"]) - np.array([0.0, 0.0, r["r2_pw_z"]])))
            r["r2_dev_rel"] = (r["r2_dev"] / abs(r["r2_pw_z"])
                               if abs(r["r2_pw_z"]) > 0 else r["r2_dev"])
            r.pop("r_all")
            rows.append(r)
        res[tag] = rows
        res[tag + "_scaling"] = dict(
            r0=powerlaw([r["beam"]["eps"] for r in rows], [r["r0_abs"] for r in rows]),
            r1=powerlaw([r["beam"]["eps"] for r in rows], [r["r1_abs"] for r in rows]),
            r2dev=powerlaw([r["beam"]["eps"] for r in rows],
                           [r["r2_dev_rel"] for r in rows]),
            du_perp=powerlaw([r["beam"]["eps"] for r in rows],
                             [r["du_perp"] for r in rows]))
    res["wall_s"] = time.perf_counter() - t0
    return res


def stage_order(cfg):
    """Systematics: field truncation order and the c0 gauge of the eps^2 correction."""
    t0 = time.perf_counter()
    rows = []
    for eps in cfg["order_eps"]:
        for (order, c0) in ((3, 0.0), (3, 1.0), (1, 0.0)):
            bm = Beam(eps=eps, a0=1.0, delta=1.0, phi0=0.3, N=1.0, order=order, c0=c0)
            r = anomaly_expansion(bm, gamma=10.0, h=cfg["h"], rtol=cfg["rtol"],
                                  atol=cfg["atol"])
            rows.append(dict(eps=eps, order=order, c0=c0, r0=r["r0"], r1=r["r1"],
                             r2=r["r2"], r1_abs=r["r1_abs"], r0_abs=r["r0_abs"],
                             du_perp=r["du_perp"]))
            rows[-1].pop("r_all", None)
    return dict(rows=rows, wall_s=time.perf_counter() - t0)


def stage_cep(cfg):
    """CEP scan: is the CEP dependence restored, and how big is it?"""
    t0 = time.perf_counter()
    res = {}
    nphi = cfg["n_phi"]
    phis = 2.0 * math.pi * np.arange(nphi) / nphi
    for tag, (gam, a0, dl, eps, N) in cfg["cep_cases"].items():
        rows = []
        for p in phis:
            bm = Beam(eps=eps, a0=a0, delta=dl, phi0=float(p), N=N)
            r = anomaly_expansion(bm, gamma=gam, h=cfg["h"], rtol=cfg["rtol"],
                                  atol=cfg["atol"])
            rows.append(dict(phi0=float(p), r0=r["r0"], r1=r["r1"], r2=r["r2"],
                             r0_abs=r["r0_abs"], r1_abs=r["r1_abs"],
                             du_perp=r["du_perp"], area=r["area"]))
        out = dict(rows=rows, gamma=gam, a0=a0, delta=dl, eps=eps, N=N)
        for key in ("r0", "r1", "r2"):
            V = np.array([row[key] for row in rows])          # (nphi, 3)
            F = np.fft.rfft(V, axis=0) / nphi
            out[key + "_mean"] = V.mean(axis=0).tolist()
            out[key + "_ptp_half"] = (0.5 * (V.max(axis=0) - V.min(axis=0))).tolist()
            out[key + "_A1"] = (2.0 * np.abs(F[1])).tolist()
            out[key + "_A2"] = (2.0 * np.abs(F[2])).tolist()
            nrm = np.linalg.norm(V, axis=1)
            out[key + "_abs_mean"] = float(nrm.mean())
            out[key + "_abs_ptp_half"] = float(0.5 * (nrm.max() - nrm.min()))
            out[key + "_rel_cep"] = float(0.5 * (nrm.max() - nrm.min())
                                          / max(nrm.mean(), 1e-300))
        res[tag] = out
    res["wall_s"] = time.perf_counter() - t0
    return res


def cep_amplitude(eps, a0, gamma, delta, N, b=0.0, n_phi=4, h=0.05,
                  rtol=1e-11, atol=1e-13):
    """Harmonic content of r0, r1, r2 in the CEP, for one configuration.

    For linear polarisation the CEP dependence turns out to be a pure first
    harmonic (the second harmonic is at the 1e-15 level in the 16-point scan of
    stage `cep`), so n_phi = 4 already resolves it exactly.
    """
    phis = 2.0 * math.pi * np.arange(n_phi) / n_phi
    V = {k: [] for k in ("r0", "r1", "r2")}
    dup = []
    for p in phis:
        bm = Beam(eps=eps, a0=a0, delta=delta, phi0=float(p), N=N)
        r = anomaly_expansion(bm, gamma=gamma, h=h, b=b, rtol=rtol, atol=atol)
        for k in V:
            V[k].append(r[k])
        dup.append(r["du_perp"])
    out = dict(eps=eps, a0=a0, gamma=gamma, delta=delta, N=N, b_over_w0=b,
               n_phi=n_phi, du_perp=float(np.mean(dup)),
               area=Beam(eps=eps, a0=a0, delta=delta, N=N).swept_area())
    for k, v in V.items():
        A = np.array(v)
        F = np.fft.rfft(A, axis=0) / n_phi
        nrm = np.linalg.norm(A, axis=1)
        out[k + "_mean"] = A.mean(axis=0).tolist()
        out[k + "_A1"] = (2.0 * np.abs(F[1])).tolist()
        out[k + "_A1_abs"] = float(np.linalg.norm(2.0 * np.abs(F[1])))
        out[k + "_abs_mean"] = float(nrm.mean())
        out[k + "_abs_ptp_half"] = float(0.5 * (nrm.max() - nrm.min()))
        out[k + "_rel_cep"] = float(out[k + "_abs_ptp_half"]
                                    / max(nrm.mean(), 1e-300))
    return out


def stage_cepscan(cfg):
    """How the CEP-odd amplitude scales.  Linear polarisation on axis (the case
    with the largest CEP modulation) plus circular polarisation OFF axis (where
    the rotational symmetry that protects circular polarisation is broken)."""
    t0 = time.perf_counter()
    e0, a00, g0, _, N0 = cfg["scan_base"]
    kw = dict(h=cfg["h"], rtol=cfg["rtol"], atol=cfg["atol"])
    res = {}
    res["lin_eps"] = [cep_amplitude(e, a00, g0, 0.0, N0, **kw)
                      for e in cfg["eps_list"]]
    res["lin_a0"] = [cep_amplitude(e0, a, g0, 0.0, N0, **kw)
                     for a in cfg["a0_list"]]
    res["lin_N"] = [cep_amplitude(e0, a00, g0, 0.0, n, **kw)
                    for n in cfg["N_list"]]
    res["lin_gamma"] = [cep_amplitude(e0, a00, g, 0.0, N0, **kw)
                        for g in cfg["gamma_list"]]
    res["lin_b"] = [cep_amplitude(e0, a00, g0, 0.0, N0, b=bb / e0, **kw)
                    for bb in cfg["b_list"]]
    for r, bb in zip(res["lin_b"], cfg["b_list"]):
        r["b_over_w0"] = bb
    res["cir_b"] = [cep_amplitude(e0, a00, g0, 1.0, N0, b=bb / e0, n_phi=8, **kw)
                    for bb in cfg["b_list"]]
    for r, bb in zip(res["cir_b"], cfg["b_list"]):
        r["b_over_w0"] = bb
    for key, xs in (("lin_eps", cfg["eps_list"]), ("lin_a0", cfg["a0_list"]),
                    ("lin_N", cfg["N_list"]), ("lin_gamma", cfg["gamma_list"])):
        res[key + "_scaling"] = dict(
            r0=powerlaw(xs, [r["r0_A1_abs"] for r in res[key]]),
            r1=powerlaw(xs, [r["r1_A1_abs"] for r in res[key]]))
    res["wall_s"] = time.perf_counter() - t0
    return res


def stage_scans(cfg):
    """a0, gamma, N, impact parameter, polarisation."""
    t0 = time.perf_counter()
    res = {}
    base = cfg["scan_base"]                       # (eps, a0, gamma, delta, N)

    def run(eps, a0, gam, dl, N, b=0.0):
        bm = Beam(eps=eps, a0=a0, delta=dl, phi0=cfg["phi0"], N=N)
        r = anomaly_expansion(bm, gamma=gam, h=cfg["h"], b=b, rtol=cfg["rtol"],
                              atol=cfg["atol"])
        r.pop("r_all")
        r["beam"] = bm.as_dict()
        return r

    e0, a00, g0, d0, N0 = base
    res["a0"] = [run(e0, a, g0, d0, N0) for a in cfg["a0_list"]]
    res["a0_scaling"] = dict(
        r0=powerlaw(cfg["a0_list"], [r["r0_abs"] for r in res["a0"]]),
        r1=powerlaw(cfg["a0_list"], [r["r1_abs"] for r in res["a0"]]),
        r2=powerlaw(cfg["a0_list"], [r["r2_abs"] for r in res["a0"]]))
    res["gamma"] = [run(e0, a00, g, d0, N0) for g in cfg["gamma_list"]]
    res["N"] = [run(e0, a00, g0, d0, n) for n in cfg["N_list"]]
    res["N_scaling"] = dict(
        r0=powerlaw(cfg["N_list"], [r["r0_abs"] for r in res["N"]]),
        r1=powerlaw(cfg["N_list"], [r["r1_abs"] for r in res["N"]]))
    res["b"] = [run(e0, a00, g0, d0, N0, b=bb / e0) for bb in cfg["b_list"]]
    for r, bb in zip(res["b"], cfg["b_list"]):
        r["b_over_w0"] = bb
    res["delta"] = [run(e0, a00, g0, d, N0) for d in cfg["delta_list"]]
    res["wall_s"] = time.perf_counter() - t0
    return res


def stage_xcheck(cfg):
    """Cross-check at the PHYSICAL anomaly a = 1.1597e-3.

    The finite-difference expansion is built at h = 0.05, i.e. 43 times the physical
    anomaly.  Here the trajectory is run with a = ANOM directly and the measured
    r(ANOM) - r(0) is compared with (i) the extrapolation ANOM*r1 + ANOM^2*r2 and
    (ii) the plane-wave holonomy -(1/2) ANOM^2 A z^, which is what the theorem
    predicts and which the focused result must approach as eps -> 0.
    """
    t0 = time.perf_counter()
    rows = []
    for (eps, dl, gam, a0, N) in cfg["xcheck_cases"]:
        bm = Beam(eps=eps, a0=a0, delta=dl, phi0=cfg["phi0"], N=N)
        tr = trajectory(bm, gamma=gam, anoms=(0.0, ANOM), rtol=1e-12, atol=1e-14)
        r = np.array(tr["r"])
        dr = r[1] - r[0]
        ex = anomaly_expansion(bm, gamma=gam, h=cfg["h"], rtol=1e-12, atol=1e-14)
        pred = ANOM * np.array(ex["r1"]) + ANOM ** 2 * np.array(ex["r2"])
        pw = np.array([0.0, 0.0, -0.5 * ANOM ** 2 * bm.swept_area()])
        rows.append(dict(eps=eps, delta=dl, gamma=gam, a0=a0, N=N,
                         dr=dr.tolist(), pred=pred.tolist(), pw=pw.tolist(),
                         dr_abs=float(np.linalg.norm(dr)),
                         pred_abs=float(np.linalg.norm(pred)),
                         pw_abs=float(np.linalg.norm(pw)),
                         rel_err_vs_pred=float(np.linalg.norm(dr - pred)
                                               / max(np.linalg.norm(pred), 1e-300)),
                         ratio_to_pw=float(np.linalg.norm(dr)
                                           / max(np.linalg.norm(pw), 1e-300)),
                         r0_abs=float(np.linalg.norm(r[0]))))
    return dict(rows=rows, anomaly=ANOM, wall_s=time.perf_counter() - t0)


def stage_a0big(cfg):
    """Intensity dependence of the CEP-driven rotation up to a0 = 24 (linear pol.).

    The low-a0 scan of stage `cepscan` is not enough: the CEP amplitude is linear in
    a0 only up to a0 ~ 2, then goes through a minimum and grows steeply.  This stage
    covers the experimentally relevant range and checks that the eps^2 law survives
    at large a0.
    """
    t0 = time.perf_counter()
    kw = dict(h=cfg["h"], rtol=cfg["rtol"], atol=cfg["atol"])
    res = {}
    res["a0"] = [cep_amplitude(0.15, a, 10.0, 0.0, 1.0, **kw)
                 for a in cfg["a0big_list"]]
    res["a0_scaling"] = dict(
        r0=powerlaw(cfg["a0big_list"], [r["r0_A1_abs"] for r in res["a0"]]),
        r1=powerlaw(cfg["a0big_list"], [r["r1_A1_abs"] for r in res["a0"]]))
    res["eps_at_a08"] = [cep_amplitude(e, 8.0, 10.0, 0.0, 1.0, **kw)
                         for e in (0.20, 0.15, 0.10, 0.07)]
    res["eps_at_a08_scaling"] = dict(
        r0=powerlaw([0.20, 0.15, 0.10, 0.07],
                    [r["r0_A1_abs"] for r in res["eps_at_a08"]]))
    res["gamma_at_a08"] = [cep_amplitude(0.15, 8.0, g, 0.0, 1.0, **kw)
                           for g in (1.0, 10.0, 1000.0)]
    res["wall_s"] = time.perf_counter() - t0
    return res


def stage_window(cfg):
    """Applicability window -- ANALYTIC ESTIMATES ONLY, nothing is simulated here.

      chi   = (hbar omega/mc^2) lam a0,  lam = k.u = gamma(1+beta)   (head-on)
      R_C   = (2/3) alpha a0 chi N                      classical RR energy fraction
              [Di Piazza, Mueller, Hatsagortsyan, Keitel, RMP 84, 1177 (2012)]
      N_gam ~ alpha a0 N                                photons emitted
      P_sf  ~ alpha a0 chi^2 N                          radiative spin flip
              [Seipt, Del Sorbo, Ridgers, Thomas, PRA 98, 023417 (2018)]
    In a focused beam chi is NOT conserved (k.u is not), so chi is evaluated with
    the initial lam and, as an upper bound, with the maximum |u| reached.
    """
    w = HBAR_OMEGA_EV / MC2_EV

    def line(gamma, a0, N, eps):
        lam = gamma + math.sqrt(max(gamma ** 2 - 1.0, 0.0))
        chi = w * lam * a0
        return dict(gamma=gamma, a0=a0, N=N, eps=eps, w0_over_lam=1.0 / (2 * math.pi * eps),
                    lam=lam, chi=chi,
                    R_C=(2.0 / 3.0) * ALPHA * a0 * chi * N,
                    N_gamma=ALPHA * a0 * N,
                    P_radflip=ALPHA * a0 * chi ** 2 * N,
                    I_Wcm2=1.37e18 * a0 ** 2 * (0.8) ** -2)
    rows = [line(g, a0, N, eps)
            for g, a0, N, eps in cfg["window_points"]]
    return dict(rows=rows, hbar_omega_eV=HBAR_OMEGA_EV,
                note="analytic estimates only; no radiation is simulated")


# ================================================================ main
def default_cfg(args):
    return dict(
        rtol=args.rtol, atol=args.atol, h=args.h, N=args.N, phi0=args.phi0,
        n_phi=args.n_phi,
        field_eps=[float(x) for x in args.field_eps.split(",")],
        conv_cases=[(0.15, 1.0, 10.0, 1.0), (0.05, 2.0, 1.0, 1.0)],
        eps_list=[float(x) for x in args.eps_list.split(",")],
        eps_cases={
            "g10_a1_cir": (10.0, 1.0, 1.0),
            "g10_a1_lin": (10.0, 1.0, 0.0),
            "g1_a1_cir": (1.0, 1.0, 1.0),
            "g1000_a1_cir": (1000.0, 1.0, 1.0),
        },
        order_eps=[0.20, 0.10, 0.05],
        cep_cases={
            "cir_e015": (10.0, 1.0, 1.0, 0.15, 1.0),
            "lin_e015": (10.0, 1.0, 0.0, 0.15, 1.0),
            "cir_e015_N05": (10.0, 1.0, 1.0, 0.15, 0.5),
        },
        scan_base=(0.15, 1.0, 10.0, 1.0, 1.0),
        a0_list=[float(x) for x in args.a0_list.split(",")],
        gamma_list=[float(x) for x in args.gamma_list.split(",")],
        N_list=[float(x) for x in args.N_list.split(",")],
        b_list=[float(x) for x in args.b_list.split(",")],
        delta_list=[0.0, 0.25, 0.5, 1.0],
        a0big_list=[float(x) for x in args.a0big_list.split(",")],
        xcheck_cases=[(0.025, 1.0, 10.0, 1.0, 1.0), (0.05, 1.0, 10.0, 1.0, 1.0),
                      (0.15, 1.0, 10.0, 1.0, 1.0), (0.15, 0.0, 10.0, 1.0, 1.0)],
        window_points=[(10.0, 1.0, 1.0, 0.15), (10.0, 10.0, 1.0, 0.15),
                       (1e3, 1.0, 1.0, 0.15), (1e3, 10.0, 1.0, 0.15),
                       (1.0, 10.0, 1.0, 0.15), (1.0, 50.0, 8.0, 0.15),
                       (1e3, 50.0, 8.0, 0.15), (1e2, 10.0, 1.0, 0.10)],
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="")
    ap.add_argument("--merge", default="")
    ap.add_argument("--out", default="focused_beam.json")
    ap.add_argument("--rtol", type=float, default=1e-11)
    ap.add_argument("--atol", type=float, default=1e-13)
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--N", type=float, default=1.0)
    ap.add_argument("--phi0", type=float, default=0.0)
    ap.add_argument("--n-phi", type=int, default=16)
    ap.add_argument("--eps-list",
                    default="0.30,0.20,0.15,0.10,0.07,0.05,0.035,0.025")
    # The three lists below used to be hard-coded.  They are exposed so that
    # `reproduce_all.py --quick` can shrink the `field`, `scans` and `a0big`
    # stages instead of skipping them; every default reproduces the published
    # run exactly.
    ap.add_argument("--field-eps",
                    default="0.30,0.20,0.15,0.10,0.07,0.05,0.035,0.025")
    ap.add_argument("--a0-list", default="0.25,0.5,1,2,4")
    ap.add_argument("--a0big-list", default="1,2,3,4,6,8,12,16,24")
    ap.add_argument("--gamma-list", default="1,3,10,100,1000")
    ap.add_argument("--N-list", default="0.5,1,2,4")
    ap.add_argument("--b-list", default="0,0.25,0.5,1,1.5")
    args = ap.parse_args(argv)
    np.random.seed(SEED)
    cfg = default_cfg(args)

    if args.merge:
        res = {}
        for fn in args.merge.split(","):
            with open(fn.strip(), encoding="utf-8") as fh:
                res.update(json.load(fh))
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print("merged ->", args.out)
        return res

    res = dict(meta=dict(script="focused_beam.py", anomaly=ANOM, seed=SEED, cfg=cfg,
                         python=sys.version.split()[0], numpy=np.__version__,
                         scipy=scipy.__version__, sympy=sp.__version__,
                         os=platform.platform(), sigma_per_N=SIG_PER_N, nsig=NSIG))
    fns = dict(field=stage_field, conv=stage_conv, eps=stage_eps, order=stage_order,
               cep=stage_cep, cepscan=stage_cepscan, scans=stage_scans,
               window=stage_window, xcheck=stage_xcheck,
               a0big=stage_a0big)
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
