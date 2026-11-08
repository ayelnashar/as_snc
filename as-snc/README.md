# AS-SNC: a spectral active-set method for SOCPs with shared norm coupling

This repository contains the implementation and the reproduction material for the article

> A. Elnashar, "A Spectral Active-Set Method for Second-Order Cone Programs with Shared Norm Coupling," submitted.

AS-SNC solves convex quadratic programs whose worst-case linear constraints share the Euclidean norm of the decision vector:

```
minimize    w' Q w
subject to  L[:, i]' w  >=  nu[i] * ||w||  +  beta[i],   i = 1, ..., N
```

with `Q` symmetric positive definite and `nu >= 0`. The method works in the eigenbasis of `Q`, reduces each working-set system to a small multiplier system and two scalar equations, and accepts a point only after a scaled KKT residual test on the original problem. If the spectral path does not pass that test, it falls back to a conic solve (Clarabel, then SCS).

## Contents

| Path | Description |
|---|---|
| `as_snc.py` | The AS-SNC solver (`as_snc`) and the residual report (`quality`). Real-valued data only. |
| `examples/quickstart.py` | Solves one random instance and prints the residual report. |
| `notebooks/AS_SNC_Supplementary_Notebook.ipynb` | Complete notebook: problem generator, embedded Fama–French data, all benchmark experiments, figures and tables. Stored outputs are those of the archived run. |
| `results/` | Archived run: JSON records for every experiment, the environment record and the figure files. |

## Installation

```
pip install -r requirements.txt
```

The benchmark compares against SCS, ECOS, Clarabel, MOSEK and Gurobi through CVXPY. MOSEK needs a licence. Gurobi's free licence limits model size, which is why some Gurobi entries are reported as size-limited. The solver itself needs only NumPy and SciPy, with CVXPY, Clarabel and SCS for the fallback.

## Usage

```python
from as_snc import as_snc

w, outer_iterations, active_set, info = as_snc(Q, L, nu, beta, tol=1e-9, return_info=True)
print(info["method"], info["residual"])
```

`info["method"]` is `"AS-SNC"` when the spectral path returns the point and `"fallback:CLARABEL"` or `"fallback:SCS"` when the conic route does. `info` also holds the multipliers and the primal, stationarity and complementarity residuals. If neither route passes the residual test, the solver raises `RuntimeError`.

## Reproducing the paper

Open the notebook (locally or in Google Colab) and run all cells. The table below links the paper to the notebook outputs.

| Paper | Notebook output |
|---|---|
| Algorithm 1 | `as_snc` |
| Example 1 | counterexample check in the additional numerical checks |
| Table 2, Figure 1 | dimension sweep, `scaling.json`, `fig_scaling.png` |
| Figure 2 | `fig_convergence.png` |
| Figure 3 | `fig_scale_invariance.png` |
| Figure 4 | `fig_active_count.png` |
| Figure 5 | tolerance sweep, `pareto.json`, `fig_pareto.png` |
| Figure 6 | conditioning sweep, `conditioning.json`, `fig_conditioning.png` |
| Figure 7 | synthetic portfolio, `portfolio.json`, `fig_portfolio.png` |
| Table 3, Figure 8 | Fama–French benchmark, `famafrench.json`, `fig_famafrench.png` |
| Table 4 | solver-only timing, `solver_only.json` |

The archived timings were measured on a Linux x86-64 cloud virtual machine with Python 3.12.14, NumPy 2.3.5, CVXPY 1.9.3 and MOSEK 11.2.6, using one BLAS thread (`results/environment.json`). Timings on other machines will differ; the accuracy results should not.

## Data

The Fama–French portfolio returns come from the Kenneth R. French Data Library, through the `linearmodels` package, and are embedded in the notebook. All other instances are synthetic and fully determined by the generator and the seeds in the notebook.

## Licence

MIT. See `LICENSE`.
