"""Test the HOLONOMY claim: M(eta) d eta = -zhat ^ d alpha  =>  the net rotation depends
ONLY on the closed curve traced by (a_x,a_y) in the polarization plane, not on how fast it
is traced. Reparametrize eta -> s(eta) (monotone, s(+-inf)=+-inf) and check Theta is unchanged.
"""
import numpy as np
from indep_check import run, boostu, area, dot

kvec = np.array([1.0, 0.0, 0.0, -1.0])
A_ANOM = 1.0 / 137.035999084 / (2 * np.pi)
u0 = boostu(1.0000001, [0, 0, 1])


def base_curve(a0=2.0, eps=0.6, sigma=4.0, cep=0.3):
    def al(s):
        env = np.exp(-s * s / (2 * sigma ** 2))
        return np.array([a0 * env * np.cos(s + cep), a0 * eps * env * np.sin(s + cep)])
    return al


def reparam(al0, kind):
    if kind == "id":
        s = lambda e: e
        sp = lambda e: 1.0 + 0 * e
    elif kind == "wiggle":                       # monotone: s' = 1+0.6cos(e) > 0
        s = lambda e: e + 0.6 * np.sin(e)
        sp = lambda e: 1.0 + 0.6 * np.cos(e)
    elif kind == "stretch":                      # slow it down by 1.7x
        s = lambda e: e / 1.7
        sp = lambda e: 1.0 / 1.7 + 0 * e
    elif kind == "tanhwarp":
        s = lambda e: e + 2.0 * np.tanh(e / 3.0)
        sp = lambda e: 1.0 + (2.0 / 3.0) / np.cosh(e / 3.0) ** 2
    else:
        raise ValueError(kind)

    def al(e):
        return al0(s(e))

    def alp(e):
        h = 1e-6
        return (al0(s(e + h)) - al0(s(e - h))) / (2 * h)
    return al, alp


def vee(R):
    Ash = 0.5 * (R - R.T)
    v = np.array([Ash[2, 1], Ash[0, 2], Ash[1, 0]])
    return np.linalg.norm(v), v


print("Reparametrization invariance of the net rotation (a = alpha/2pi):")
al0 = base_curve()
for kind, L in (("id", 40.0), ("wiggle", 40.0), ("stretch", 70.0), ("tanhwarp", 45.0)):
    al, alp = reparam(al0, kind)
    A = area(al, alp, L)
    uf, R, kap, _ = run(al, alp, A_ANOM, u0, kvec, L, rtol=1e-12, atol=1e-14)
    ang, v = vee(R)
    print(f"  {kind:9s} A={A:.8f}  Theta_num={ang:.9e}  (1/2)a^2A={0.5*A_ANOM**2*A:.9e}"
          f"  axis_z={v[2]/ang:+.4f}")

print()
print("Same but with a LARGE anomaly (a=0.25) -> tests all orders, not just O(a^2):")
for kind, L in (("id", 40.0), ("wiggle", 40.0), ("stretch", 70.0), ("tanhwarp", 45.0)):
    al, alp = reparam(al0, kind)
    A = area(al, alp, L)
    uf, R, kap, _ = run(al, alp, 0.25, u0, kvec, L, rtol=1e-12, atol=1e-14)
    ang, v = vee(R)
    print(f"  {kind:9s} A={A:.8f}  Theta_num={ang:.9e}  (1/2)a^2A={0.5*0.25**2*A:.9e}"
          f"  axis_z={v[2]/ang:+.4f}")

print()
print("CEP independence for exactly circular polarization (rigid rotation of the curve):")
for cep in (0.0, 0.5, 1.3, 2.7):
    alc = base_curve(a0=2.0, eps=1.0, sigma=4.0, cep=cep)
    al, alp = reparam(alc, "id")
    A = area(al, alp, 40.0)
    uf, R, kap, _ = run(al, alp, 0.25, u0, kvec, 40.0, rtol=1e-12, atol=1e-14)
    ang, v = vee(R)
    print(f"  cep={cep:.2f} A={A:.8f} Theta(a=0.25)={ang:.10e} axis_z={v[2]/ang:+.6f}")

print()
print("CEP dependence for ELLIPTIC (delta=0.6) at large anomaly (path shape changes):")
for cep in (0.0, 0.5, 1.3, 2.7):
    ale = base_curve(a0=2.0, eps=0.6, sigma=4.0, cep=cep)
    al, alp = reparam(ale, "id")
    A = area(al, alp, 40.0)
    uf, R, kap, _ = run(al, alp, 0.25, u0, kvec, 40.0, rtol=1e-12, atol=1e-14)
    ang, v = vee(R)
    print(f"  cep={cep:.2f} A={A:.8f} Theta(a=0.25)={ang:.10e} (1/2)a^2A={0.5*0.0625*A:.6e}")
