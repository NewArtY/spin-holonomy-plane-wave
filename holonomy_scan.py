# -*- coding: utf-8 -*-
r"""
holonomy_scan.py -- the data set behind FIG. 2 of
"Net electron spin rotation in a plane-wave pulse: Holonomy set by the
anomalous magnetic moment" (Physical Review D).

QUESTION
--------
Equation (10) of the main text states that the net rest-frame spin rotation of
an electron that has crossed a finite plane-wave pulse is a holonomy,

    Theta_net = -(1/2) a_e^2 A + O(a_e^4),
    A         = int (a_x a_y' - a_y a_x') deta          (signed swept area),

with a_e = (g-2)/2 the anomalous moment and (a_x, a_y)(eta) the transverse
normalised potential.  The claim is that NOTHING else enters: not the envelope,
not the number of cycles, not the chirp, not the carrier-envelope phase, not the
electron energy, and not the rate at which the curve C: eta -> (a_x, a_y) is
traced out.  This script produces one point per pulse configuration in the plane
(A, Theta_net) so that the collapse onto the straight line can be seen directly,
and it measures the two families for which A vanishes identically.

It is a thin driver on top of spin_magnitude.py, which contains the validated
plane-wave machinery (exact Volkov orbit, covariant T-BMT right-hand side,
DOP853 transport of an orthonormal rest-frame triad, rotation vector read off
the resulting SO(3) matrix).  Nothing of that is re-implemented here; this file
only adds the pulse shapes that FIG. 2 needs and that spin_magnitude.Pulse does
not provide.

PULSE SHAPES ADDED HERE
-----------------------
ChirpedPulse   a_x = s f(eta) cos(psi), a_y = s d f(eta) sin(psi),
               psi = eta + phi0 + c eta^2/(2 sigma).  Linear chirp: the curve C
               is traced out at a phase-dependent rate, which is precisely the
               reparameterisation the holonomy is supposed to ignore.
GatedPulse     polarisation gating, i.e. two counter-rotating circular pulses of
               amplitude ratio q, delayed by Delta:
                   a_x = s (f_+ + q f_-) cos(eta+phi0),
                   a_y = s (f_+ - q f_-) sin(eta+phi0),  f_+- = f(eta +- Delta/2).
               The instantaneous ellipticity sweeps from +1 through 0 to -1
               inside one pulse, so C is a non-convex, self-intersecting curve.
               At q = 1 the two helicities carry equal weight and the signed
               area cancels identically, A = (1-q^2) int f^2 deta + O(e^{-2s^2});
               the scan therefore uses q < 1.
FigureEight    a_x = a0 f(eta),  a_y = c a0 f(eta)^2  (no carrier).  The curve is
               a parabolic arc traced out and back, so
                   a_x a_y' - a_y a_x' = c a_x^2 a_x' = (c/3) (a_x^3)'
               integrates to zero EXACTLY.  This is the second null family of
               FIG. 2: A = 0 with a curve that is not a straight segment, hence a
               test of the area law and not of the linear-polarisation argument.
FlatTop        circular polarisation, |a_perp| = a0 on a flat top of length
               2 eta_flat with cos^2 ramps of length L_r on each side.  This is
               the geometry for which Eq. (12) of the main text,
                   Theta_exact = (sqrt(1 + a_e^2 a0^2) - 1) Delta_eta,
               resums the whole a_e-series; it feeds the inset of FIG. 2.

All shapes obey a_perp(+-eta_max) = 0 to machine precision, which is the only
condition Maxwell imposes on a finite pulse.

STAGES
------
main    ~90 configurations at anomaly a_e = 1e-2: Gaussian and cos^2 envelopes,
        ellipticities |delta| = 0.05..1 of both helicities, a0 = 1..10,
        N = 0.5..16 cycles, gamma = 1..1e4, eight carrier-envelope phases,
        four chirps, four gate delays.  Records (A, Theta_net) and the residual
        against Eq. (10).
null    the two families with A = 0: linear polarisation (both envelopes,
        a0 = 1, 10, N = 1, 4) and the figure-eight curve (c = 0.5, 1, 2,
        a0 = 1, 3, 10), each at anomalies up to a_e = 0.6.  Records the measured
        |Theta_net|, i.e. the round-off floor.
inset   flat-top circular pulse, anomaly scanned so that x = a_e a0 runs from
        0.02 to 0.9; records Theta_net/(-(1/2) a_e^2 A) against x together with
        the resummation (sqrt(1+x^2)-1)/(x^2/2) of Eq. (12).
wrap    the one place where the angle read from the matrix is NOT the holonomy:
        a four-cycle circular plateau at a0 = 3 and a_e = 0.2, where the
        accumulated angle passes pi and the extracted angle folds back.  This
        is the worked case quoted in Sec. S2.3 of the Supplemental Material.

Output: holonomy_scan.json (consumed by make_fig1.py).

Usage
    python holonomy_scan.py --stages main,null,inset,wrap --out holonomy_scan.json
    python holonomy_scan.py --stages main,null,inset,wrap --quick --out quick.json

Runtime: ~50 s for the full set on one core (measured, i7 laptop); ~10 s with
--quick.  Single process, no threads.
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
from scipy.integrate import quad

from spin_magnitude import (ANOM, N_SIGMA_TRUNC, SEED, SIG_PER_N, L_PER_N,
                            Pulse, net_rotation)

A_FIG1 = 1.0e-2          # anomaly used for the main panel of FIG. 2
                         # (the figure was FIG. 1 up to deposit v1.0.1)
RTOL = 1.0e-13
ATOL = 1.0e-16


# --------------------------------------------------------------- pulse shapes
class _Base:
    """Common services: numerical swept area and endpoint check."""

    def swept_area(self):
        def integ(e):
            ax, ay = self.a(e)
            dx, dy = self.da(e)
            return float(ax * dy - ay * dx)
        return quad(integ, -self.eta_max, self.eta_max, limit=1000)[0]

    def endpoint_a(self):
        lo, hi = self.a(-self.eta_max), self.a(self.eta_max)
        return float(max(abs(lo[0]), abs(lo[1]), abs(hi[0]), abs(hi[1])))


class ChirpedPulse(_Base):
    """Gaussian envelope, elliptical polarisation, linear chirp."""

    def __init__(self, a0=1.0, N=1.0, phi0=0.0, delta=1.0, chirp=0.0):
        self.a0, self.N, self.phi0 = float(a0), float(N), float(phi0)
        self.delta, self.chirp = float(delta), float(chirp)
        self.sigma = SIG_PER_N * self.N
        self.eta_max = N_SIGMA_TRUNC * self.sigma
        self.norm = 1.0 / math.sqrt(1.0 + self.delta ** 2)

    def _psi(self, e):
        return e + self.phi0 + self.chirp * e ** 2 / (2.0 * self.sigma)

    def _dpsi(self, e):
        return 1.0 + self.chirp * e / self.sigma

    def f(self, e):
        return np.exp(-np.asarray(e, float) ** 2 / (2.0 * self.sigma ** 2))

    def df(self, e):
        e = np.asarray(e, float)
        return -e / self.sigma ** 2 * self.f(e)

    def a(self, e):
        s = self.a0 * self.norm
        f, p = self.f(e), self._psi(e)
        return s * f * np.cos(p), s * self.delta * f * np.sin(p)

    def da(self, e):
        s = self.a0 * self.norm
        f, df, p, dp = self.f(e), self.df(e), self._psi(e), self._dpsi(e)
        return (s * (df * np.cos(p) - f * np.sin(p) * dp),
                s * self.delta * (df * np.sin(p) + f * np.cos(p) * dp))

    def as_dict(self):
        return dict(shape="chirp", a0=self.a0, N=self.N, phi0=self.phi0,
                    delta=self.delta, chirp=self.chirp, env="gauss",
                    sigma=self.sigma, eta_max=self.eta_max)


class GatedPulse(_Base):
    """Polarisation gating: two counter-rotating circular pulses, delay Delta."""

    def __init__(self, a0=1.0, N=1.0, phi0=0.0, gate=1.0, q=0.6):
        self.a0, self.N, self.phi0 = float(a0), float(N), float(phi0)
        self.gate, self.q = float(gate), float(q)    # Delta / sigma, amplitude ratio
        self.sigma = SIG_PER_N * self.N
        self.d = self.gate * self.sigma
        self.eta_max = N_SIGMA_TRUNC * self.sigma + abs(self.d)

    def _fpm(self, e):
        e = np.asarray(e, float)
        s2 = 2.0 * self.sigma ** 2
        fp = np.exp(-(e + 0.5 * self.d) ** 2 / s2)
        fm = np.exp(-(e - 0.5 * self.d) ** 2 / s2)
        return fp, fm

    def _dfpm(self, e):
        e = np.asarray(e, float)
        fp, fm = self._fpm(e)
        return (-(e + 0.5 * self.d) / self.sigma ** 2 * fp,
                -(e - 0.5 * self.d) / self.sigma ** 2 * fm)

    def a(self, e):
        fp, fm = self._fpm(e)
        c, s = np.cos(e + self.phi0), np.sin(e + self.phi0)
        h, q = 0.5 * self.a0, self.q
        return h * (fp + q * fm) * c, h * (fp - q * fm) * s

    def da(self, e):
        fp, fm = self._fpm(e)
        dfp, dfm = self._dfpm(e)
        c, s = np.cos(e + self.phi0), np.sin(e + self.phi0)
        h, q = 0.5 * self.a0, self.q
        return (h * ((dfp + q * dfm) * c - (fp + q * fm) * s),
                h * ((dfp - q * dfm) * s + (fp - q * fm) * c))

    def as_dict(self):
        return dict(shape="gate", a0=self.a0, N=self.N, phi0=self.phi0,
                    gate=self.gate, q=self.q, env="gauss", sigma=self.sigma,
                    eta_max=self.eta_max)


class FigureEight(_Base):
    """a_y proportional to a_x^2: a parabolic arc traced out and back, A = 0."""

    def __init__(self, a0=1.0, N=1.0, curv=1.0):
        self.a0, self.N, self.curv = float(a0), float(N), float(curv)
        self.sigma = SIG_PER_N * self.N
        self.eta_max = N_SIGMA_TRUNC * self.sigma

    def f(self, e):
        return np.exp(-np.asarray(e, float) ** 2 / (2.0 * self.sigma ** 2))

    def df(self, e):
        e = np.asarray(e, float)
        return -e / self.sigma ** 2 * self.f(e)

    def a(self, e):
        f = self.f(e)
        return self.a0 * f, self.curv * self.a0 * f ** 2

    def da(self, e):
        f, df = self.f(e), self.df(e)
        return self.a0 * df, 2.0 * self.curv * self.a0 * f * df

    def as_dict(self):
        return dict(shape="fig8", a0=self.a0, N=self.N, curv=self.curv,
                    env="gauss", sigma=self.sigma, eta_max=self.eta_max)


class FlatTop(_Base):
    """Circular polarisation, |a_perp| = a0 on the flat top, cos^2 ramps."""

    def __init__(self, a0=1.0, eta_flat=math.pi, L_r=math.pi / 2.0, phi0=0.0):
        self.a0, self.phi0 = float(a0), float(phi0)
        self.eta_flat, self.L_r = float(eta_flat), float(L_r)
        self.eta_max = self.eta_flat + self.L_r

    def f(self, e):
        x = np.abs(np.asarray(e, float))
        w = math.pi / (2.0 * self.L_r)
        out = np.where(x <= self.eta_flat, 1.0, 0.0)
        m = (x > self.eta_flat) & (x < self.eta_max)
        return np.where(m, np.cos(w * (x - self.eta_flat)) ** 2, out)

    def df(self, e):
        x = np.asarray(e, float)
        sg, ax = np.sign(x), np.abs(x)
        w = math.pi / (2.0 * self.L_r)
        m = (ax > self.eta_flat) & (ax < self.eta_max)
        d = -2.0 * w * np.cos(w * (ax - self.eta_flat)) * np.sin(
            w * (ax - self.eta_flat)) * sg
        return np.where(m, d, 0.0)

    def a(self, e):
        f = self.f(e)
        return (self.a0 * f * np.cos(e + self.phi0),
                self.a0 * f * np.sin(e + self.phi0))

    def da(self, e):
        f, df = self.f(e), self.df(e)
        c, s = np.cos(e + self.phi0), np.sin(e + self.phi0)
        return self.a0 * (df * c - f * s), self.a0 * (df * s + f * c)

    def as_dict(self):
        return dict(shape="flattop", a0=self.a0, phi0=self.phi0,
                    eta_flat=self.eta_flat, L_r=self.L_r, env="flattop",
                    eta_max=self.eta_max)


# --------------------------------------------------------------- measurement
def measure(pulse, gamma=1.0, anom=A_FIG1, family="", label="",
            rtol=RTOL, atol=ATOL):
    """One configuration: signed area, net rotation, residual against Eq. (10)."""
    area = pulse.swept_area()
    r = net_rotation(gamma, pulse, anom=anom, rtol=rtol, atol=atol)
    pred = -0.5 * anom ** 2 * area
    rec = dict(family=family, label=label, gamma=float(gamma), anom=float(anom),
               area=float(area), theta=float(r["theta"]),
               rx=float(r["rx"]), ry=float(r["ry"]), rz=float(r["rz"]),
               pred=float(pred),
               resid=float(r["rz"] - pred),
               rel_resid=(float((r["rz"] - pred) / pred) if pred != 0.0
                          else None),
               orth_err=float(r["orth_err"]), du=float(r["du"]),
               su_err=float(r["su_err"]), ss_err=float(r["ss_err"]),
               nfev=int(r["nfev"]), end_a=pulse.endpoint_a())
    rec.update({("p_" + k): v for k, v in pulse.as_dict().items()})
    return rec


# --------------------------------------------------------------- stages
def stage_main(quick=False):
    """(A, Theta_net) for every pulse family, at the single anomaly a_e = 1e-2."""
    t0 = time.perf_counter()
    rows = []
    a0s = (1.0, 3.0, 10.0) if not quick else (1.0, 10.0)

    # (a) Gaussian, ellipticity of both helicities
    deltas = ((0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0, -0.05, -0.2, -0.5, -1.0)
              if not quick else (0.2, 1.0, -0.5))
    for d in deltas:
        for a0 in a0s:
            pu = Pulse(a0=a0, N=1.0, phi0=0.0, env="gauss", delta=d)
            rows.append(measure(pu, family="gauss", label=f"d={d:g},a0={a0:g}"))

    # (b) cos^2 envelope
    for d in ((0.2, 0.5, 1.0) if not quick else (1.0,)):
        for a0 in a0s:
            for N in ((1.0, 4.0) if not quick else (1.0,)):
                pu = Pulse(a0=a0, N=N, phi0=0.0, env="cos2", delta=d)
                rows.append(measure(pu, family="cos2",
                                    label=f"d={d:g},a0={a0:g},N={N:g}"))

    # (c) pulse length, circular
    for N in ((0.5, 1.0, 2.0, 4.0, 8.0, 16.0) if not quick else (0.5, 4.0)):
        pu = Pulse(a0=1.0, N=N, phi0=0.0, env="gauss", delta=1.0)
        rows.append(measure(pu, family="Nscan", label=f"N={N:g}"))

    # (d) Lorentz factor
    for g in ((1.0, 10.0, 1e2, 1e3, 1e4) if not quick else (1.0, 1e3)):
        pu = Pulse(a0=1.0, N=1.0, phi0=0.0, env="gauss", delta=1.0)
        rows.append(measure(pu, gamma=g, family="gamma", label=f"g={g:g}"))

    # (e) carrier-envelope phase over [0, 2pi)
    n_phi = 8 if not quick else 3
    for k in range(n_phi):
        p = 2.0 * math.pi * k / n_phi
        pu = Pulse(a0=3.0, N=1.0, phi0=p, env="gauss", delta=0.5)
        rows.append(measure(pu, family="cep", label=f"phi0={p:.3f}"))

    # (f) linear chirp
    for c in ((-0.10, -0.05, 0.05, 0.10) if not quick else (0.10,)):
        for d in ((0.5, 1.0) if not quick else (1.0,)):
            pu = ChirpedPulse(a0=3.0, N=2.0, phi0=0.0, delta=d, chirp=c)
            rows.append(measure(pu, family="chirp", label=f"c={c:g},d={d:g}"))

    # (g) polarisation gating (amplitude ratio q < 1, otherwise A cancels)
    for gt in ((0.5, 1.0, 1.5, 2.0) if not quick else (1.0,)):
        for q in ((0.4, 0.7) if not quick else (0.4,)):
            pu = GatedPulse(a0=3.0, N=2.0, phi0=0.0, gate=gt, q=q)
            rows.append(measure(pu, family="gate", label=f"D/s={gt:g},q={q:g}"))

    rel = [abs(r["rel_resid"]) for r in rows if r["rel_resid"] is not None]
    return dict(anom=A_FIG1, rows=rows, n=len(rows),
                max_rel_resid=max(rel), median_rel_resid=float(np.median(rel)),
                max_end_a=max(r["end_a"] for r in rows),
                max_orth=max(r["orth_err"] for r in rows),
                wall_s=time.perf_counter() - t0)


def stage_null(quick=False):
    """The two families with A = 0, at anomalies up to 0.6."""
    t0 = time.perf_counter()
    lin, f8 = [], []
    anoms = (1e-2, 1e-1, 0.6) if not quick else (0.6,)
    for env in (("gauss", "cos2") if not quick else ("gauss",)):
        for a0 in ((1.0, 10.0) if not quick else (10.0,)):
            for N in ((1.0, 4.0) if not quick else (1.0,)):
                for A in anoms:
                    pu = Pulse(a0=a0, N=N, phi0=0.3, env=env, delta=0.0)
                    lin.append(measure(pu, anom=A, family="linear",
                                       label=f"{env},a0={a0:g},N={N:g},a={A:g}"))
    for c in ((0.5, 1.0, 2.0) if not quick else (1.0,)):
        for a0 in ((1.0, 3.0, 10.0) if not quick else (10.0,)):
            for A in ((1e-2, 0.6) if not quick else (0.6,)):
                pu = FigureEight(a0=a0, N=1.0, curv=c)
                f8.append(measure(pu, anom=A, family="fig8",
                                  label=f"c={c:g},a0={a0:g},a={A:g}"))
    return dict(linear=lin, fig8=f8,
                max_theta_linear=max(r["theta"] for r in lin),
                max_theta_fig8=max(r["theta"] for r in f8),
                max_area_fig8=max(abs(r["area"]) for r in f8),
                wall_s=time.perf_counter() - t0)


def stage_inset(quick=False):
    """Flat-top circular pulse: ratio to the area law against x = a_e a0."""
    t0 = time.perf_counter()
    pu = FlatTop(a0=1.0, eta_flat=math.pi, L_r=math.pi / 2.0)
    area = pu.swept_area()
    xs = ((0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
          if not quick else (0.02, 0.3, 0.9))
    rows = []
    for x in xs:
        r = measure(pu, anom=x, family="flattop", label=f"x={x:g}")
        eq7 = (math.sqrt(1.0 + x * x) - 1.0) / (0.5 * x * x)
        r.update(x=x, ratio=r["rz"] / r["pred"], eq7=eq7,
                 eq7_dev=r["rz"] / r["pred"] / eq7 - 1.0)
        rows.append(r)
    return dict(rows=rows, area=float(area),
                delta_eta_eff=float(area / pu.a0 ** 2),
                max_dev_from_eq7=max(abs(r["eq7_dev"]) for r in rows),
                wall_s=time.perf_counter() - t0)


def stage_wrap(quick=False):
    r"""The angle read from an SO(3) matrix wraps; the holonomy does not.

    Section S2.3 of the Supplemental Material defines Theta_net as the
    CONTINUOUS LIFT in the anomaly from a_e = 0, i.e. the angle accumulated
    along the path, whereas the quantity an SO(3) matrix hands back lies in
    [0, pi], is defined modulo 2 pi, and carries a sign ambiguity that goes with
    a reversal of the axis.  The two disagree as soon as the accumulated angle
    passes pi, and the Supplemental quotes one worked case of that.  Nothing in
    this deposit produced it until now; this stage does.

    The case: a four-cycle circular plateau (eta_flat = 4 pi, i.e. a flat top of
    total length 8 pi) with one-cycle cos^2 fronts (L_r = 2 pi), a0 = 3,
    a_e = 0.2, head-on at gamma = 1.

    Three numbers are produced and they are not the same number:
      * the leading-order area law, -(1/2) a_e^2 A;
      * the resummation for a circular plateau,
        (sqrt(1 + a_e^2 a0^2) - 1) Delta_eta with Delta_eta = A/a0^2, which is
        the expression that is accurate here because a_e a0 = 0.6 is not small;
      * the angle actually read from the matrix.
    The stage also sweeps the anomaly so that the fold at Theta = pi is visible
    as a fold and not as an anomaly of the data.

    NOTE FOR THE MANUSCRIPT: the prediction quoted in Sec. S2.3 as "4.96 rad"
    is the RESUMMATION, not the leading-order area law, which gives 5.37 rad
    for this pulse.  The measured 1.307 rad is 2 pi minus 4.98 rad.
    """
    t0 = time.perf_counter()
    a0, anom = 3.0, 0.2
    pu = FlatTop(a0=a0, eta_flat=4.0 * math.pi, L_r=2.0 * math.pi)
    area = pu.swept_area()
    deta = area / a0 ** 2
    rec = measure(pu, gamma=1.0, anom=anom, family="wrap", label="4-cycle plateau")
    lead = -0.5 * anom ** 2 * area
    resum = -(math.sqrt(1.0 + (anom * a0) ** 2) - 1.0) * deta
    meas = rec["theta"]
    lifted = 2.0 * math.pi - meas             # the branch the lift selects here
    rec.update(area=area, delta_eta=deta,
               pred_leading=lead, pred_resummed=resum,
               theta_measured=meas, rz_measured=rec["rz"],
               lifted_from_matrix=lifted,
               axis_sign_flipped=bool(rec["rz"] * resum < 0.0),
               lifted_vs_resummed_rel=abs(lifted / abs(resum) - 1.0),
               lifted_vs_leading_rel=abs(lifted / abs(lead) - 1.0))

    sweep = []
    anoms = ((0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.2, 0.24, 0.28, 0.32)
             if not quick else (0.05, 0.2, 0.32))
    for A in anoms:
        r = measure(pu, gamma=1.0, anom=A, family="wrap", label=f"a_e={A:g}")
        de = area / a0 ** 2
        res = (math.sqrt(1.0 + (A * a0) ** 2) - 1.0) * de
        sweep.append(dict(anom=A, theta=r["theta"], rz=r["rz"],
                          pred_leading=0.5 * A ** 2 * area,
                          pred_resummed=res,
                          folded=bool(res > math.pi),
                          reduced_pred=float(abs((res + math.pi) % (2.0 * math.pi)
                                                 - math.pi)),
                          orth_err=r["orth_err"]))
    first_fold = next((s["anom"] for s in sweep if s["folded"]), None)
    return dict(case=rec, sweep=sweep, a0=a0, anom=anom,
                eta_flat=4.0 * math.pi, L_r=2.0 * math.pi,
                first_folded_anomaly=first_fold,
                max_reduced_rel_err=max(
                    abs(s["theta"] / s["reduced_pred"] - 1.0)
                    for s in sweep if s["reduced_pred"] > 1e-3),
                wall_s=time.perf_counter() - t0)


# --------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stages", default="main,null,inset,wrap")
    ap.add_argument("--out", default="holonomy_scan.json")
    ap.add_argument("--quick", action="store_true",
                    help="reduced grids, ~10x faster, same code paths")
    args = ap.parse_args(argv)

    np.random.seed(SEED)
    res = dict(meta=dict(script="holonomy_scan.py", anomaly_physical=ANOM,
                         anomaly_fig1=A_FIG1, seed=SEED, quick=bool(args.quick),
                         rtol=RTOL, atol=ATOL, python=sys.version.split()[0],
                         numpy=np.__version__, scipy=scipy.__version__,
                         os=platform.platform(), sigma_per_N=SIG_PER_N,
                         L_per_N=L_PER_N, n_sigma_trunc=N_SIGMA_TRUNC))
    fns = dict(main=stage_main, null=stage_null, inset=stage_inset,
               wrap=stage_wrap)
    for s in [x.strip() for x in args.stages.split(",") if x.strip()]:
        t = time.perf_counter()
        res[s] = fns[s](quick=args.quick)
        print(f"[{s}] done in {time.perf_counter() - t:.1f} s", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("written", args.out, flush=True)
    return res


if __name__ == "__main__":
    main()
