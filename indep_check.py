"""
INDEPENDENT verification of the "no net spin rotation in a plane wave" theorem.
Written from scratch, without looking at bench/spin_magnitude.py.

Conventions
-----------
metric g = diag(+1,-1,-1,-1)
wave 4-vector      k^mu   (k^2 = 0),   phase eta = k.x
normalized potential  alpha^mu(eta) = e A^mu(eta) / m ,  k.alpha = 0, alpha purely transverse
field              f^{mu nu} = (e/m) F^{mu nu} = k^mu alpha'^nu - k^nu alpha'^mu
Lorentz force      du^mu/dtau = f^{mu nu} u_nu
T-BMT              dS^mu/dtau = (1+a) f^{mu nu} S_nu + a u^mu (S_lam f^{lam nu} u_nu),  a=(g-2)/2
kappa = k.u = const;  d/dtau = kappa d/deta
"""
import numpy as np
from scipy.integrate import solve_ivp

G = np.diag([1.0, -1.0, -1.0, -1.0])


def dot(x, y):
    return x @ G @ y


def wedge(p, q):
    """(p^q)^mu_nu = p^mu q_nu - q^mu p_nu  (matrix acting on S^nu)."""
    return np.outer(p, G @ q) - np.outer(q, G @ p)


# ---------------------------------------------------------------- pulses
def make_pulse(kind="ellip", a0=1.0, eps=0.5, sigma=4.0, cep=0.0, n=1.0):
    """returns alpha(eta) -> (a1,a2) and alphap(eta)."""
    if kind in ("ellip", "lin", "circ"):
        e2 = {"lin": 0.0, "circ": 1.0}.get(kind, eps)

        def al(e):
            env = np.exp(-e * e / (2 * sigma ** 2))
            return np.array([a0 * env * np.cos(n * e + cep),
                             a0 * e2 * env * np.sin(n * e + cep)])

        def alp(e):
            env = np.exp(-e * e / (2 * sigma ** 2))
            de = -e / sigma ** 2 * env
            return np.array([a0 * (de * np.cos(n * e + cep) - env * n * np.sin(n * e + cep)),
                             a0 * e2 * (de * np.sin(n * e + cep) + env * n * np.cos(n * e + cep))])
    elif kind == "dc":            # unipolar vector potential: int alpha deta != 0
        def al(e):
            env = np.exp(-e * e / (2 * sigma ** 2))
            return np.array([a0 * env, a0 * eps * env * np.cos(n * e + cep)])

        def alp(e):
            env = np.exp(-e * e / (2 * sigma ** 2))
            de = -e / sigma ** 2 * env
            return np.array([a0 * de,
                             a0 * eps * (de * np.cos(n * e + cep) - env * n * np.sin(n * e + cep))])
    elif kind == "rotplane":      # linear polarization whose plane rotates with eta
        def th(e):
            return 0.6 * np.tanh(e / sigma)

        def al(e):
            env = np.exp(-e * e / (2 * sigma ** 2))
            c = a0 * env * np.cos(n * e + cep)
            return np.array([c * np.cos(th(e)), c * np.sin(th(e))])

        def alp(e):
            h = 1e-6
            return (al(e + h) - al(e - h)) / (2 * h)
    elif kind == "broken":        # alpha(+inf) != alpha(-inf): violates the hypothesis
        def al(e):
            return np.array([a0 * (np.tanh(e / sigma) + 1.0) / 2.0, 0.0 * e])

        def alp(e):
            return np.array([a0 / (2 * sigma) / np.cosh(e / sigma) ** 2, 0.0 * e])
    elif kind == "fig8":          # closed curve with zero enclosed area
        def al(e):
            env = np.exp(-e * e / (2 * sigma ** 2))
            return np.array([a0 * env * np.cos(n * e), a0 * eps * env * np.sin(2 * n * e)])

        def alp(e):
            h = 1e-6
            return (al(e + h) - al(e - h)) / (2 * h)
    else:
        raise ValueError(kind)
    return al, alp


def area(al, alp, L):
    """calligraphic A = int (a_x a_y' - a_y a_x') deta"""
    from scipy.integrate import quad
    f = lambda e: al(e)[0] * alp(e)[1] - al(e)[1] * alp(e)[0]
    v, _ = quad(f, -L, L, limit=400)
    return v


# ---------------------------------------------------------------- dynamics
def run(al, alp, anom, u0, kvec, L=40.0, rtol=1e-12, atol=1e-14, spins=None):
    kappa = dot(kvec, u0)

    def A4(e):
        a1, a2 = al(e)
        return np.array([0.0, a1, a2, 0.0])

    def Ap4(e):
        a1, a2 = alp(e)
        return np.array([0.0, a1, a2, 0.0])

    def rhs(e, y):
        u = y[:4]
        f = wedge(kvec, Ap4(e))          # f^mu_nu
        w = f @ u                        # w^mu = f^{mu nu} u_nu
        Om = (1.0 + anom) * f + anom * wedge(u, w)
        du = f @ u / kappa
        out = [du]
        for i in range(1, len(y) // 4):
            out.append(Om @ y[4 * i:4 * i + 4] / kappa)
        return np.concatenate(out)

    # orthonormal rest-frame triad of u0, boosted to the lab
    gam = u0[0]
    bet = u0[1:] / gam
    b2 = bet @ bet
    triad = []
    for i in range(3):
        sp = np.zeros(3)
        sp[i] = 1.0
        s0 = gam * (bet @ sp)
        sv = sp + (gam - 1.0) * (bet @ sp) * bet / b2 if b2 > 1e-30 else sp
        triad.append(np.concatenate(([s0], sv)))
    if spins is not None:
        triad = spins
    y0 = np.concatenate([u0] + triad)
    sol = solve_ivp(rhs, (-L, L), y0, rtol=rtol, atol=atol, method="DOP853",
                    dense_output=False)
    yf = sol.y[:, -1]
    uf = yf[:4]
    Sf = [yf[4 * (i + 1):4 * (i + 2)] for i in range(3)]
    # net map in the rest-frame triad basis:  R_ij = <triad_i, S_j(final)>  (spatial metric)
    R = np.zeros((3, 3))
    for j in range(3):
        for i in range(3):
            R[i, j] = -dot(triad[i], Sf[j])   # -g(.,.) = euclidean in rest frame
    return uf, R, kappa, sol


def rot_angle_axis(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(ax)
    if n > 1e-300:
        ax = ax / n
    return th, ax, np.sign(np.trace(R) - 1.0)


def signed_angle_about(R, axis):
    """signed rotation angle of R about a given unit axis (assumes R is a rotation about it)."""
    # build a vector perpendicular to axis
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(tmp @ axis) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = tmp - (tmp @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    v = R @ e1
    return np.arctan2(v @ e2, v @ e1)


def boostu(gamma, beta_dir):
    b = np.sqrt(1.0 - 1.0 / gamma ** 2)
    v = np.array(beta_dir, dtype=float)
    v = v / np.linalg.norm(v) * b
    return np.concatenate(([gamma], gamma * v))


RESULTS = []          # filled by report(); harvested by indep_check_all.py


def report(name, R, extra=""):
    dev = np.linalg.norm(R - np.eye(3))
    th, ax, _ = rot_angle_axis(R)
    print(f"{name:52s} |R-1|={dev:.3e}  angle={th:.6e}  axis=({ax[0]:+.3f},{ax[1]:+.3f},{ax[2]:+.3f}) {extra}")
    RESULTS.append(dict(name=name.strip(), dev=float(dev), angle=float(th),
                        axis=[float(v) for v in ax], extra=extra.strip()))
    return dev, th, ax


def main_t1():
    """T1: g = 2 exactly -> the net map is the identity for every pulse and u0."""
    np.set_printoptions(precision=6, suppress=False)
    kvec = np.array([1.0, 0.0, 0.0, -1.0])     # wave propagates along -z (head-on for +z electron)
    L = 40.0
    print("=" * 110)
    print("T1  g = 2 exactly (anomaly a=0): net map must be IDENTITY for any pulse / u0")
    for kind in ("lin", "ellip", "circ", "dc", "rotplane", "fig8"):
        al, alp = make_pulse(kind, a0=2.0, eps=0.6, sigma=4.0, cep=0.7)
        for gam, bd in ((1.0000001, [0, 0, 1]), (50.0, [0, 0, 1]), (20.0, [0.3, 0.1, 1.0])):
            u0 = boostu(gam, bd)
            uf, R, kap, _ = run(al, alp, 0.0, u0, kvec, L)
            report(f"  a=0 {kind:9s} gamma={gam:8.3f} u0perp={u0[1]:+.3f}", R,
                   f"|u-u0|={np.linalg.norm(uf-u0):.2e}")


if __name__ == "__main__":
    main_t1()
