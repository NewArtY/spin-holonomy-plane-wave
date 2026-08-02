import numpy as np
from indep_check import make_pulse, run, boostu, area, dot, G

kvec = np.array([1.0, 0.0, 0.0, -1.0])
L = 40.0
A_ANOM = 1.0 / 137.035999084 / (2 * np.pi)


def vee(R):
    """well-conditioned small-angle extraction: axial vector of the antisymmetric part"""
    Ash = 0.5 * (R - R.T)
    v = np.array([Ash[2, 1], Ash[0, 2], Ash[1, 0]])
    n = np.linalg.norm(v)
    return n, (v / n if n > 0 else v)


print("=" * 110)
print("T9  transverse initial momentum: angle must be INVARIANT (null rotation conjugation),")
print("    axis = propagation direction of the wave AS SEEN IN THE ELECTRON REST FRAME")
al, alp = make_pulse("ellip", a0=2.0, eps=0.6, sigma=4.0, cep=0.3)
A = area(al, alp, L)
print(f"   A = {A:.8f}   (1/2)a^2 A = {0.5*A_ANOM**2*A:.8e}")
for gam, bd in ((1.0000001, [0, 0, 1]), (20.0, [0, 0, 1]), (20.0, [0.4, 0.2, 1.0]),
                (20.0, [1.5, -0.7, 1.0]), (5.0, [2.0, 0.0, 0.3]), (20.0, [0, 0, -1.0])):
    u0 = boostu(gam, bd)
    uf, R, kap, _ = run(al, alp, A_ANOM, u0, kvec, L)
    ang, ax = vee(R)
    # propagation direction of k in the rest frame of u0, in the triad basis
    khat = kvec - dot(kvec, u0) * u0          # component of k orthogonal to u0
    nn = np.sqrt(-dot(khat, khat))
    khat = khat / nn
    tri = []
    bet = u0[1:] / u0[0]
    b2 = bet @ bet
    for i in range(3):
        sp = np.zeros(3); sp[i] = 1.0
        s0 = u0[0] * (bet @ sp)
        sv = sp + (u0[0] - 1.0) * (bet @ sp) * bet / b2 if b2 > 1e-30 else sp
        tri.append(np.concatenate(([s0], sv)))
    krest = np.array([-dot(tri[i], khat) for i in range(3)])
    print(f"   gamma={gam:8.3f} u0perp=({u0[1]:+.2f},{u0[2]:+.2f}) kappa={kap:8.3f}  angle={ang:.8e}"
          f"  axis=({ax[0]:+.4f},{ax[1]:+.4f},{ax[2]:+.4f})  k_restframe=({krest[0]:+.4f},{krest[1]:+.4f},{krest[2]:+.4f})")

print("=" * 110)
print("T10  zero-area 2D pulse (alpha_y = c alpha_x^2, curve retraced -> A == 0):")
print("     expect Theta = 0 at O(a^2); check residual scaling in a")


def zeroarea(a0=2.0, c=0.5, sigma=3.0, n=1.0):
    def al(e):
        x = a0 * np.exp(-e * e / (2 * sigma ** 2)) * np.cos(n * e)
        return np.array([x, c * x * x])

    def alp(e):
        h = 1e-5
        return (al(e + h) - al(e - h)) / (2 * h)
    return al, alp


alz, alpz = zeroarea()
Az = area(alz, alpz, L)
print(f"   A(numeric) = {Az:.3e}")
u0 = boostu(1.0000001, [0, 0, 1])
prev = None
for anom in (0.03, 0.1, 0.3, 0.6):
    uf, R, kap, _ = run(alz, alpz, anom, u0, kvec, L, rtol=1e-12, atol=1e-14)
    ang, ax = vee(R)
    r = "" if prev is None else f"   ratio vs previous a: {ang/prev[0]:.3f}  (a-ratio^3={ (anom/prev[1])**3:.3f}, ^4={(anom/prev[1])**4:.3f})"
    print(f"   a={anom:6.3f}  |angle|={ang:.6e}  axis=({ax[0]:+.3f},{ax[1]:+.3f},{ax[2]:+.3f}){r}")
    prev = (ang, anom)

print("=" * 110)
print("T11  exact circular resummation, small a, long flat top")


def flat_circ(a0, ncyc, ramp=8.0):
    T = 2 * np.pi * ncyc

    def env(e):
        x = abs(e)
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


for a0, anom in ((1.0, 0.05), (1.0, 0.3), (3.0, 0.1), (3.0, 0.3), (10.0, 0.05)):
    al2, alp2, T = flat_circ(a0, 2.0)
    Lc = T / 2 + 9.0
    A2 = area(al2, alp2, Lc)
    Deta = A2 / a0 ** 2
    uf, R, kap, _ = run(al2, alp2, anom, u0, kvec, Lc, rtol=1e-12, atol=1e-14)
    ang, ax = vee(R)
    print(f"   a0={a0:5.1f} a={anom:5.3f} a*a0={anom*a0:6.3f}  |angle|_num={ang:.6e}"
          f"  (1/2)a^2A={0.5*anom**2*A2:.6e}  resum=(sqrt(1+a^2a0^2)-1)*Deta={ (np.sqrt(1+anom**2*a0**2)-1)*Deta:.6e}"
          f"  axis_z={ax[2]:+.4f}")

print("=" * 110)
print("T12  chirped / few-cycle / arbitrary shapes, g=2 and linear pol at large anomaly")


def chirp(a0=2.5, sigma=5.0, c=0.15, eps=0.0, cep=1.1):
    def ph(e):
        return e + c * e * e

    def al(e):
        env = np.exp(-e * e / (2 * sigma ** 2))
        return np.array([a0 * env * np.cos(ph(e) + cep), a0 * eps * env * np.sin(ph(e) + cep)])

    def alp(e):
        h = 1e-6
        return (al(e + h) - al(e - h)) / (2 * h)
    return al, alp


for eps in (0.0, 0.45):
    alc, alpc = chirp(eps=eps)
    Ac = area(alc, alpc, L)
    for anom, gam in ((0.0, 500.0), (A_ANOM, 1.0000001), (A_ANOM, 500.0), (0.3, 30.0)):
        u0 = boostu(gam, [0, 0, 1])
        uf, R, kap, _ = run(alc, alpc, anom, u0, kvec, L, rtol=1e-12, atol=1e-14)
        ang, ax = vee(R)
        print(f"   chirped eps={eps:.2f} a={anom:8.5f} gamma={gam:7.1f} A={Ac:+10.6f}"
              f"  |angle|={ang:.6e}  (1/2)a^2A={0.5*anom**2*Ac:+.6e} axis_z={ax[2]:+.4f}")
