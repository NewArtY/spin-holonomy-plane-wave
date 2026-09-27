# -*- coding: utf-8 -*-
r"""
rr_ikt_check.py -- two estimates of Sec. VII and Sec. S7 of
"Net electron spin rotation in a plane-wave pulse: Holonomy set by the
anomalous magnetic moment" (Physical Review D).

1. RADIATION REACTION (stage `rr`)
----------------------------------
Classical radiation reaction in the Landau-Lifshitz form, with the spin
Fermi-Walker transported along the non-electromagnetic acceleration:

    du/dtau = f u + a_RR,
    dS/dtau = (1 + a_e) f S + a_e B(u, f u) S + B(a_RR, u) S,
    a_RR    = tau0 [ (df/dtau) u + f f u + ((f u).(f u)) u ].

The last two terms of a_RR (the "leading" term) equal
tau0 kappa |a_perp'|^2 (k - kappa u).  In the interaction picture of Sec. III B
they generate a boost along the wave with a fixed axis, so they leave the
anomalous holonomy unchanged at every order in a_e; they only add a rotation
about an in-plane axis equal to the net transverse kick.  The first term
(Schott term) adds, at second order in tau0, a g-independent rotation about
nhat of -(1/2) (tau0 kappa)^2 A', where A' is the signed area of the curve
traced by a_perp' (the hodograph); for the Gaussian circular pulse used here
A' = A [1 + 3/(2 sigma^2)].  The script checks both statements with tau0
inflated by five orders of magnitude, for a circularly polarized Gaussian
pulse, electron initially at rest (kappa = 1), and repeats the Schott check
for an electron moving against the wave with kappa = 3.

2. FIELD-DRESSED ANOMALY (stage `ikt`)
-------------------------------------
If the anomaly depends on the local field, a_e -> a_e + Delta a_e(eta), the
first-order term  int Delta a_e(eta) a_perp'(eta) d eta  survives.  For a
carrier-dominated pulse the integrand is antiperiodic in the carrier phase
(a_perp' changes sign under a half-period shift, the field strength does not),
so the integral averages out over each period.  The script evaluates
|int d(eta) a_perp' d eta| / (rho d_max), rho = max |a_perp|, for d
proportional to the field strength and to its square, for circular, linear and
elliptical (delta = 0.5) pulses of N = 0.5 to 4 cycles, maximized over the
carrier-envelope phase, and compares it with the bound 2 used in the text.

Units: k = (1,0,0,1) (unit frequency), eta = t - z, metric (+,-,-,-).
Output: rr_ikt_check.json and rr_ikt_check.log.

Usage
    python rr_ikt_check.py --out rr_ikt_check.json --log rr_ikt_check.log
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

G = np.diag([1.0, -1.0, -1.0, -1.0])
K = np.array([1.0, 0.0, 0.0, 1.0])
SIG_PER_N = math.pi / math.sqrt(math.log(2.0))


def bv(p, q):
    """B(p,q)^mu_nu = p^mu q_nu - q^mu p_nu."""
    return np.outer(p, G @ q) - np.outer(q, G @ p)


def mdot(p, q):
    return float(p @ G @ q)


def circular_pulse(rho, N):
    s = SIG_PER_N * N

    def fields(e):
        h = math.exp(-e * e / (2 * s * s))
        hp = -e / (s * s) * h
        hpp = (e * e / s**4 - 1 / (s * s)) * h
        z = complex(math.cos(e), math.sin(e))
        a = rho * h * z
        a1 = rho * (hp + 1j * h) * z
        a2 = rho * (hpp + 2j * hp - h) * z
        vec = lambda c: np.array([0.0, c.real, c.imag, 0.0])
        return vec(a), vec(a1), vec(a2)
    area = rho**2 * s * math.sqrt(math.pi)
    return fields, 8 * s, area


def hodograph_area(rho, N):
    """Signed area functional of a_perp': int (a_x' a_y'' - a_y' a_x'') d eta."""
    s = SIG_PER_N * N
    return rho**2 * s * math.sqrt(math.pi) * (1 + 1.5 / (s * s))


def pure_boost(u):
    """Pure boost taking (1,0,0,0) to u."""
    gam = u[0]
    v = u[1:] / gam
    b = np.eye(4)
    if np.linalg.norm(v) > 0:
        n = v / np.linalg.norm(v)
        b[0, 0] = gam
        b[0, 1:] = gam * v
        b[1:, 0] = gam * v
        b[1:, 1:] = np.eye(3) + (gam - 1) * np.outer(n, n)
    return b


def net_rotation(ae, tau0, mode, rho, N, kappa0=1.0, rtol=1e-11, atol=1e-13):
    """Rotation vector of the spin relative to the momentum, rest frame of u0.

    kappa0 = k.u0 > 1 means an electron moving against the wave along -z."""
    fields, emax, area = circular_pulse(rho, N)

    def rhs(e, y):
        u = y[:4]
        S = y[4:].reshape(3, 4)
        _, a1, a2 = fields(e)
        f = bv(K, a1)
        kap = mdot(K, u)
        fu = f @ u
        aRR = np.zeros(4)
        if tau0:
            if mode in ("full", "leading"):
                aRR += tau0 * (f @ fu + mdot(fu, fu) * u)
            if mode in ("full", "schott"):
                aRR += tau0 * kap * (bv(K, a2) @ u)
        gen = (1 + ae) * f + ae * bv(u, fu) + bv(aRR, u)
        return np.concatenate([(fu + aRR) / kap, (S @ gen.T).ravel() / kap])

    # u0 = (gamma, 0, 0, -gamma beta) with gamma (1 + beta) = kappa0
    g0 = (kappa0**2 + 1) / (2 * kappa0)
    u0 = np.array([g0, 0.0, 0.0, -(kappa0**2 - 1) / (2 * kappa0)])
    y0 = np.concatenate([u0, (pure_boost(u0) @ np.eye(4)[:, 1:]).T.ravel()])
    sol = solve_ivp(rhs, (-emax, emax), y0, method="DOP853", rtol=rtol, atol=atol)
    uf = sol.y[:4, -1]
    Sf = sol.y[4:, -1].reshape(3, 4)
    Sb = (np.linalg.inv(pure_boost(uf)) @ Sf.T).T
    rv = Rotation.from_matrix(Sb[:, 1:].T).as_rotvec()
    # transverse momentum change relative to u0 (u0 has no transverse part)
    return dict(rotvec=rv.tolist(), u_final=uf.tolist(),
                kappa_final=float(uf[0] - uf[3]),
                du_perp=float(math.hypot(uf[1], uf[2])), area=area)


def stage_rr(log):
    ae, rho, N = 0.05, 2.0, 2.0
    ref = net_rotation(ae, 0.0, "full", rho, N)
    law = -0.5 * ae**2 * ref["area"]
    rows = []
    log(f"[rr] circular Gaussian, rho = {rho}, N = {N}, a_e = {ae}, kappa0 = 1")
    log(f"     A = {ref['area']:.6f}, law -a_e^2 A/2 = {law:.6e}, "
        f"no RR: Theta_z = {ref['rotvec'][2]:.9e}")
    for tau0 in (1e-4, 3e-4, 1e-3):
        for mode in ("leading", "schott", "full"):
            r = net_rotation(ae, tau0, mode, rho, N)
            dz = (r["rotvec"][2] - ref["rotvec"][2]) / abs(ref["rotvec"][2])
            inpl = math.hypot(r["rotvec"][0], r["rotvec"][1])
            rows.append(dict(ae=ae, tau0=tau0, mode=mode, rel_change_z=dz,
                             in_plane=inpl, **r))
            log(f"     tau0={tau0:7.1e} {mode:7s}  rel. change of Theta_z = {dz:+.3e}  "
                f"in-plane = {inpl:.3e}  kappa_f = {r['kappa_final']:.6f}  "
                f"|du_perp| = {r['du_perp']:.3e}")
    g0 = []
    ap = hodograph_area(rho, N)
    log(f"     hodograph area A' = {ap:.6f} = A x {ap / ref['area']:.6f}")
    for kappa0 in (1.0, 3.0):
        for mode in (("leading", "schott", "full") if kappa0 == 1.0 else ("schott",)):
            tau0 = 1e-3 / kappa0          # same tau0 kappa in both runs
            r = net_rotation(0.0, tau0, mode, rho, N, kappa0=kappa0)
            pred = -0.5 * (tau0 * kappa0) ** 2 * ap
            g0.append(dict(ae=0.0, tau0=tau0, kappa0=kappa0, mode=mode,
                           schott_pred=pred, **r))
            log(f"     a_e = 0, kappa0 = {kappa0:.0f}, tau0 = {tau0:.3e}, {mode:7s}: "
                f"rotvec = ({r['rotvec'][0]:+.4e}, {r['rotvec'][1]:+.4e}, "
                f"{r['rotvec'][2]:+.4e}), |du_perp| = {r['du_perp']:.3e}, "
                f"-(1/2)(tau0 kappa)^2 A' = {pred:.4e}")
    return dict(reference=ref, law=law, hodograph_area=ap, rows=rows,
                g_independent=g0)


def stage_ikt(log):
    rows = []
    log("[ikt] max over CEP of |int d(eta) a_perp' d eta| / (rho d_max), "
        "d ~ |a_perp'|^p, rho = max |a_perp|")
    ceps = [math.pi * j / 16 for j in range(16)]   # period pi suffices
    for N in (0.5, 1.0, 2.0, 4.0):
        s = SIG_PER_N * N
        e = np.linspace(-9 * s, 9 * s, 200001)
        h = np.exp(-e**2 / (2 * s * s))
        hp = -e / (s * s) * h
        for delta, lab in ((1.0, "circular"), (0.0, "linear"), (0.5, "elliptical")):
            n = 1.0 / math.sqrt(1 + delta**2)
            for p in (1, 2):
                vals = []
                for phi0 in (ceps if delta != 1.0 else [0.0]):
                    ax = n * h * np.cos(e + phi0)
                    ay = n * delta * h * np.sin(e + phi0)
                    axp = n * (hp * np.cos(e + phi0) - h * np.sin(e + phi0))
                    ayp = n * delta * (hp * np.sin(e + phi0) + h * np.cos(e + phi0))
                    rho = float(np.max(np.hypot(ax, ay)))
                    E = np.hypot(axp, ayp)
                    d = (E / E.max()) ** p
                    vals.append(math.hypot(np.trapezoid(d * axp, e),
                                           np.trapezoid(d * ayp, e)) / rho)
                j = int(np.argmax(vals))
                even = vals[0] if delta != 1.0 else None
                rows.append(dict(N=N, pulse=lab, power=p, value=float(vals[j]),
                                 cep_at_max=(ceps[j] if delta != 1.0 else 0.0),
                                 value_even_pulse=even))
                log(f"     N={N:<4} {lab:11s} p={p}: max {vals[j]:.3e} "
                    f"(CEP {ceps[j] if delta != 1.0 else 0.0:.3f})"
                    + (f", even pulse (CEP 0) {even:.1e}" if even is not None else "")
                    + "   (bound 2)")
    mx = max(r["value"] for r in rows)
    log(f"     overall maximum {mx:.3f}; at N = 2: "
        f"{max(r['value'] for r in rows if r['N'] == 2.0):.2e}")
    return dict(rows=rows, bound=2.0, max_value=mx)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Radiation-reaction and dressed-anomaly checks.")
    ap.add_argument("--out", default="rr_ikt_check.json")
    ap.add_argument("--log", default="rr_ikt_check.log")
    args = ap.parse_args(argv)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    t0 = time.perf_counter()
    res = dict(meta=dict(script="rr_ikt_check.py", units="k=(1,0,0,1), eta=t-z"),
               rr=stage_rr(log), ikt=stage_ikt(log))
    res["meta"]["wall_s"] = time.perf_counter() - t0
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(args.log, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("written", args.out, args.log)


if __name__ == "__main__":
    main()
