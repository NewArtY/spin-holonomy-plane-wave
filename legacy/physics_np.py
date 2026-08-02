# -*- coding: utf-8 -*-
"""
physics_np.py -- TORCH-FREE definition of the model system.

Split out of common.py deliberately: on Windows, ProcessPoolExecutor uses the
'spawn' start method, so every worker re-imports the module that defines the
worker function.  If that module (transitively) imports torch, each worker pays
a ~1-3 s import and ~250-400 MB RSS.  With 8 workers that alone can wedge a
4-core/24 GB box.  bench_ivp.py therefore imports ONLY from here.

common.py re-exports these names, so existing torch-based scripts are unchanged.
"""
import numpy as np


class Params:
    def __init__(self, E0=0.50, omega=2.0 * np.pi, tau_p=0.60, phi0=0.0,
                 g=2.002319304, kT=0.60, tau_cut=2.0):
        self.E0 = E0
        self.omega = omega
        self.tau_p = tau_p
        self.phi0 = phi0
        self.g = g
        self.kT = kT
        self.tau_cut = tau_cut          # domain = [-tau_cut, +tau_cut]

    def as_dict(self):
        return dict(E0=self.E0, omega=self.omega, tau_p=self.tau_p, phi0=self.phi0,
                    g=self.g, kT=self.kT, tau_cut=self.tau_cut)


def field_np(tau, P):
    return P.E0 * np.exp(-tau ** 2 / (2.0 * P.tau_p ** 2)) * np.cos(P.omega * tau + P.phi0)


def rhs_np(tau, y, P):
    th, p, _ = y
    a = field_np(tau, P)
    dth = -np.sin(p) * (np.cos(th) + a * np.sin(th))
    dp = np.cos(p) * (np.sin(th) - a * np.cos(th))
    dSg = P.kT * np.sin(th) * np.cos(p) + (P.g - 1.0) * a * np.cos(th)
    return [dth, dp, dSg]


Y0 = np.array([0.30, 0.10, 0.0])          # (theta, p, Sigma) at tau = -tau_cut
