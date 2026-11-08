"""Solve one random SNC-SOCP instance with AS-SNC and print the residual report."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from as_snc import as_snc  # noqa: E402

rng = np.random.default_rng(0)
M, N = 50, 8
F = rng.standard_normal((M, M))
Q = F @ F.T / M + 0.1 * np.eye(M)          # symmetric positive definite cost
L = rng.standard_normal((M, N))            # constraint directions
nu = np.full(N, 0.2)                       # uncertainty radii
beta = np.ones(N)                          # offsets

w, outer, active, info = as_snc(Q, L, nu, beta, tol=1e-9, return_info=True)
print("method:", info["method"])
print("outer iterations:", outer, "| active constraints:", [int(i) for i in active])
print("objective:", info["objective"])
print("residuals: primal %.2e, stationarity %.2e, complementarity %.2e"
      % (info["primal"], info["stationarity"], info["complementarity"]))
