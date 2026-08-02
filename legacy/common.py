# -*- coding: utf-8 -*-
"""
common.py -- model system + PINN definition shared by all benchmark scripts.

Model system (same *class* as the paper's rapidity-BMT closure Eqs. (4)-(5),
but with the hyperbolic functions replaced by their bounded trigonometric
analogues so that the benchmark cannot diverge; the computational profile --
non-autonomous 1-DOF canonical pair + one driven spin angle, all coupled
through the same Gaussian-envelope field -- is identical).

    a(tau)   = E0 * exp(-tau^2 / (2 tau_p^2)) * cos(omega*tau + phi0)     [field]
    H(th,p,tau) = cos(th)cos(p) + a(tau) sin(th) cos(p)                   [cf. Eq.(S13)]

    dth/dtau = +dH/dp  = -sin(p) * ( cos(th) + a(tau) sin(th) )
    dp /dtau = -dH/dth = +cos(p) * ( sin(th) - a(tau) cos(th) )
    dSg/dtau = Omega_T + (g-1) * Omega_L
               Omega_T = kT * sin(th) * cos(p)          (Thomas-like)
               Omega_L = a(tau) * cos(th)               (Larmor-like, CEP-carrying)

Network:  N_Theta(tau) -> (theta, p, Sigma),  input width 1, tanh hidden, linear out.
Loss   :  L = L_res + lam_H * L_H + lam_sym * L_sym + L_ic          (Eq. (6) of the plan)
    L_res : autodiff residual of the three ODEs
    L_H   : dH/dtau (autodiff, through the network) - partial H/partial tau (explicit)
    L_sym : second-order consistency  d2x/dtau2 - d/dtau f(x(tau),tau)  for the
            canonical pair (structure-preservation surrogate; requires 2nd-order
            autodiff, which is what dominates the cost of the real L_sym term)
    L_ic  : initial conditions at tau = tau_min
"""
import math
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------- physics
# Params / field_np / rhs_np / Y0 live in the torch-free module physics_np.py so
# that multiprocessing workers (spawn on Windows) never import torch.
# Re-exported here for backward compatibility of the torch-based benchmarks.
from physics_np import Params, field_np, rhs_np, Y0     # noqa: F401


def field_t(tau, P):
    return P.E0 * torch.exp(-tau ** 2 / (2.0 * P.tau_p ** 2)) * torch.cos(P.omega * tau + P.phi0)


def dfield_dtau_t(tau, P):
    """d a / d tau, analytic."""
    env = P.E0 * torch.exp(-tau ** 2 / (2.0 * P.tau_p ** 2))
    ph = P.omega * tau + P.phi0
    return env * (-tau / P.tau_p ** 2) * torch.cos(ph) - env * P.omega * torch.sin(ph)


def rhs_t(tau, th, p, P):
    a = field_t(tau, P)
    dth = -torch.sin(p) * (torch.cos(th) + a * torch.sin(th))
    dp = torch.cos(p) * (torch.sin(th) - a * torch.cos(th))
    dSg = P.kT * torch.sin(th) * torch.cos(p) + (P.g - 1.0) * a * torch.cos(th)
    return dth, dp, dSg


def hamiltonian_t(tau, th, p, P):
    a = field_t(tau, P)
    return torch.cos(th) * torch.cos(p) + a * torch.sin(th) * torch.cos(p)


def dH_dtau_explicit_t(tau, th, p, P):
    return dfield_dtau_t(tau, P) * torch.sin(th) * torch.cos(p)


# ----------------------------------------------------------------------------- network
class MLP(nn.Module):
    def __init__(self, width=128, depth=5, n_in=1, n_out=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        layers = []
        d = n_in
        for _ in range(depth):
            lin = nn.Linear(d, width)
            with torch.no_grad():
                bound = math.sqrt(6.0 / (d + width))
                lin.weight.uniform_(-bound, bound, generator=g)
                lin.bias.zero_()
            layers += [lin, nn.Tanh()]
            d = width
        out = nn.Linear(d, n_out)
        with torch.no_grad():
            bound = math.sqrt(6.0 / (d + n_out))
            out.weight.uniform_(-bound, bound, generator=g)
            out.bias.zero_()
        layers += [out]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(q.numel() for q in self.parameters())


def to_phys_tau(z, P):
    """z in [-1,1] -> tau in [-tau_cut, tau_cut]"""
    return z * P.tau_cut


# ----------------------------------------------------------------------------- losses
def grad1(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True)[0]


def pinn_losses(net, z, P, use_H=True, use_sym=True, y0=None):
    """z: (N,1) collocation points in [-1,1], requires_grad=True."""
    if y0 is None:
        y0 = Y0
    s = P.tau_cut                                   # dtau/dz
    tau = to_phys_tau(z, P)
    out = net(z)
    th, p, Sg = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    dth = grad1(th, z) / s
    dp = grad1(p, z) / s
    dSg = grad1(Sg, z) / s

    f_th, f_p, f_Sg = rhs_t(tau, th, p, P)
    r_th, r_p, r_Sg = dth - f_th, dp - f_p, dSg - f_Sg
    L_res = (r_th ** 2).mean() + (r_p ** 2).mean() + (r_Sg ** 2).mean()

    L_H = torch.zeros((), dtype=z.dtype)
    if use_H:
        H = hamiltonian_t(tau, th, p, P)
        dH = grad1(H, z) / s
        L_H = ((dH - dH_dtau_explicit_t(tau, th, p, P)) ** 2).mean()

    L_sym = torch.zeros((), dtype=z.dtype)
    if use_sym:
        # second-order (variational) consistency on the canonical pair
        d2th = grad1(dth, z) / s
        d2p = grad1(dp, z) / s
        df_th = grad1(f_th, z) / s
        df_p = grad1(f_p, z) / s
        L_sym = ((d2th - df_th) ** 2).mean() + ((d2p - df_p) ** 2).mean()

    z0 = torch.full((1, 1), -1.0, dtype=z.dtype)
    o0 = net(z0)
    tgt = torch.tensor(y0, dtype=z.dtype).view(1, 3)
    L_ic = ((o0 - tgt) ** 2).sum()

    return L_res, L_H, L_sym, L_ic


# ----------------------------------------------------------------------------- memory
def peak_rss_mb():
    """Peak working set of this process, MB (Windows)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 1024 ** 2
    except Exception as e:                                    # pragma: no cover
        return float("nan")


def env_info():
    import platform
    import sys
    import scipy
    return dict(python=sys.version.split()[0], torch=torch.__version__,
                numpy=np.__version__, scipy=scipy.__version__,
                cuda=torch.cuda.is_available(), threads=torch.get_num_threads(),
                cpu=platform.processor(), machine=platform.machine(),
                os=platform.platform())
