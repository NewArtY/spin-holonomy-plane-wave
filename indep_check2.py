import numpy as np
from indep_check import (make_pulse, run, report, boostu, area, rot_angle_axis,
                         signed_angle_about, wedge, dot, G)

kvec = np.array([1.0, 0.0, 0.0, -1.0])
L = 40.0
ALPHA_QED = 1.0 / 137.035999084
A_ANOM = ALPHA_QED / (2 * np.pi)     # 1.1614e-3

print("=" * 112)
print("T2  LINEAR polarization, arbitrary anomaly a (even huge), arbitrary gamma/u0perp/CEP -> identity?")
for anom in (A_ANOM, 0.1, 1.0, -0.7):
    for cep in (0.0, 0.7, 1.9):
        al, alp = make_pulse("lin", a0=3.0, sigma=4.0, cep=cep)
        for gam, bd in ((1.0000001, [0, 0, 1]), (100.0, [0, 0, 1]), (20.0, [0.4, 0.2, 1.0])):
            u0 = boostu(gam, bd)
            uf, R, kap, _ = run(al, alp, anom, u0, kvec, L)
            report(f"  a={anom:+.4f} cep={cep:.1f} gamma={gam:8.3f} uperp={u0[1]:+.2f}", R)

print("=" * 112)
print("T3  ELLIPTIC: Theta_net vs (1/2) a^2 A, scaling in anomaly a")
al, alp = make_pulse("ellip", a0=2.0, eps=0.6, sigma=4.0, cep=0.3)
A = area(al, alp, L)
print(f"   calligraphic A = int (ax ay' - ay ax') deta = {A:.10f}")
u0 = boostu(1.0000001, [0, 0, 1])
for anom in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1):
    uf, R, kap, _ = run(al, alp, anom, u0, kvec, L)
    th, ax, _ = rot_angle_axis(R)
    thz = signed_angle_about(R, np.array([0.0, 0.0, 1.0]))
    pred = 0.5 * anom ** 2 * A
    print(f"   a={anom:9.1e}  Theta_num={thz:+.6e}  (1/2)a^2A={pred:+.6e}  ratio={thz/pred:+.8f}"
          f"  axis=({ax[0]:+.4f},{ax[1]:+.4f},{ax[2]:+.4f})")

print("=" * 112)
print("T4  gamma-independence of Theta (elliptic, a = alpha/2pi)")
for gam, bd in ((1.0000001, [0, 0, 1]), (10.0, [0, 0, 1]), (100.0, [0, 0, 1]),
                (1000.0, [0, 0, 1]), (20.0, [0.4, 0.2, 1.0])):
    u0 = boostu(gam, bd)
    uf, R, kap, _ = run(al, alp, A_ANOM, u0, kvec, L)
    # rotation axis should be the propagation direction expressed in the rest frame triad
    thz = signed_angle_about(R, np.array([0.0, 0.0, 1.0]))
    th, ax, _ = rot_angle_axis(R)
    print(f"   gamma={gam:9.3f} kappa={kap:9.3f} uperp={u0[1]:+.2f}  Theta_z={thz:+.6e}"
          f"  |angle|={th:.6e}  pred={0.5*A_ANOM**2*A:+.6e} axis=({ax[0]:+.3f},{ax[1]:+.3f},{ax[2]:+.3f})")

print("=" * 112)
print("T5  CIRCULAR flat-top: exact resummation Theta = (sqrt(1+a^2 a0^2)-1) * Deta ?")


def flat_circ(a0, ncyc, ramp=6.0):
    """flat-top circular pulse, smooth sin^2 ramps"""
    T = 2 * np.pi * ncyc

    def env(e):
        x = np.abs(e)
        if x <= T / 2:
            return 1.0
        if x >= T / 2 + ramp:
            return 0.0
        return np.cos(np.pi * (x - T / 2) / (2 * ramp)) ** 2

    def al(e):
        return np.array([a0 * env(e) * np.cos(e), a0 * env(e) * np.sin(e)])

    def alp(e):
        h = 1e-6
        return (al(e + h) - al(e - h)) / (2 * h)
    return al, alp, T


for a0 in (1.0, 5.0, 20.0):
    for anom in (A_ANOM, 0.05, 0.2):
        al2, alp2, T = flat_circ(a0, 3.0)
        Lc = T / 2 + 7.0
        A2 = area(al2, alp2, Lc)
        u0 = boostu(1.0000001, [0, 0, 1])
        uf, R, kap, _ = run(al2, alp2, anom, u0, kvec, Lc, rtol=1e-11, atol=1e-13)
        thz = signed_angle_about(R, np.array([0.0, 0.0, 1.0]))
        pred2 = 0.5 * anom ** 2 * A2
        predx = (np.sqrt(1 + anom ** 2 * a0 ** 2) - 1) * (A2 / a0 ** 2)
        print(f"   a0={a0:5.1f} a={anom:8.5f}  A={A2:10.4f}  Theta={thz:+.8e}"
              f"  (1/2)a^2A={pred2:+.8e}  exact-resum={predx:+.8e}")

print("=" * 112)
print("T6  other pulse shapes at a = alpha/2pi:  Theta vs (1/2)a^2 A")
for kind, kw in (("lin", dict(a0=3.0)), ("dc", dict(a0=2.0, eps=0.7)),
                 ("rotplane", dict(a0=2.0)), ("fig8", dict(a0=2.0, eps=0.8)),
                 ("ellip", dict(a0=2.0, eps=1.0))):
    al3, alp3 = make_pulse(kind, sigma=4.0, cep=0.4, **kw)
    A3 = area(al3, alp3, L)
    for anom in (A_ANOM, 0.1):
        u0 = boostu(1.0000001, [0, 0, 1])
        uf, R, kap, _ = run(al3, alp3, anom, u0, kvec, L)
        thz = signed_angle_about(R, np.array([0.0, 0.0, 1.0]))
        th, ax, _ = rot_angle_axis(R)
        print(f"   {kind:9s} a={anom:8.5f} A={A3:+12.6f} Theta={thz:+.6e} (1/2)a^2A={0.5*anom**2*A3:+.6e}"
              f"  |R-1|={np.linalg.norm(R-np.eye(3)):.3e} axis=({ax[0]:+.3f},{ax[1]:+.3f},{ax[2]:+.3f})")

print("=" * 112)
print("T7  hypothesis violated: alpha(+inf) != alpha(-inf)  (net momentum kick)")
al4, alp4 = make_pulse("broken", a0=2.0, sigma=4.0)
for anom in (0.0, A_ANOM, 0.1):
    u0 = boostu(20.0, [0, 0, 1])
    uf, R, kap, _ = run(al4, alp4, anom, u0, kvec, L)
    print(f"   a={anom:8.5f}  |u_f-u_0|={np.linalg.norm(uf-u0):.6e}  |R-1|={np.linalg.norm(R-np.eye(3)):.6e}")

print("=" * 112)
print("T8  interaction-picture generator:  M = Ad_{exp(-A)} Omega_1  should equal -zhat ^ alpha'  (kappa=1)")
u0 = np.array([1.0, 0, 0, 0.0])          # rest frame, kappa = k.u0 = 1
zhat = np.array([0.0, 0, 0, 1.0])
al5, alp5 = make_pulse("ellip", a0=2.0, eps=0.6, sigma=4.0, cep=0.3)
import scipy.linalg as sla
maxdev = 0.0
for e in np.linspace(-8, 8, 33):
    a1, a2 = al5(e)
    p1, p2 = alp5(e)
    A4 = np.array([0.0, a1, a2, 0.0])
    Ap4 = np.array([0.0, p1, p2, 0.0])
    f = wedge(kvec, Ap4)
    phi = (dot(A4, u0) - 0.5 * dot(A4, A4)) / 1.0
    u = u0 - A4 + phi * kvec
    w = f @ u
    Om1 = f + wedge(u, w)                 # coefficient of the anomaly a in Omega
    Amat = wedge(kvec, A4)
    M = sla.expm(-Amat) @ Om1 @ sla.expm(Amat)
    Mpred = -wedge(zhat, Ap4)
    maxdev = max(maxdev, np.max(np.abs(M - Mpred)))
print(f"   max |M - (-zhat^alpha')| over eta grid = {maxdev:.3e}")
