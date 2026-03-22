# Weighted Approximation

Numerical experiments for weighted-norm and sample-path dependent bounds on suboptimality gaps arising from MDP model approximations.

## Overview

We study an inventory MDP under two models: a true model **M** and an approximate model **M_hat** that differ in demand distribution and holding cost. The project computes and compares three types of bounds on the suboptimality gap $V^{\hat{\pi}^\star\} - V^\star$:

1. **Sup-norm bound** -- uniform constant bound over all states
2. **Weighted-norm bound** -- state-dependent bound using a Lyapunov weight function
3. **Sample-path bound** -- state-dependent bound via fixed-point iteration on local error certificates, tightest in the operating region

## Folder Structure

```
.
├── models.py                  # InventoryMDP class (parameters, demand PMF, holding cost)
├── solvers.py                 # Value iteration (H-array trick), Bellman operators, policy evaluation
├── bounds.py                  # Kappa computation, weighted-norm bounds, sample-path fixed-point bound
├── plots.py                   # Plotting utilities for zoomed-in/zoomed-out figures
├── tests.py                   # Sanity checks for models, solvers, and bounds
├── run_existing_experiments.py  # Reproduces figures from the model approximation paper
├── run_new_experiments.py       # Generates sample-path bound comparison figures
├── notebooks/
│   ├── Weighted_Approx.ipynb        # Original weighted-norm experiments
│   └── Sample_Path_Bounds.ipynb     # Sample-path bound derivation and comparison
├── figures/
│   ├── fig1-fig5/             # Figures from the model approximation paper
│   └── comparison/            # Sample-path vs existing bound comparison plots
├── docs/                      # LaTeX derivations and notes
└── scripts/                   # Auxiliary scripts (bug comparison, etc.)
```

## Sample-Path Bounds Notebook

`notebooks/Sample_Path_Bounds.ipynb` implements the sample-path dependent AIS bound (Theorem 2, MDP Corollary). It computes a state-dependent function alpha(s) via fixed-point iteration that decomposes the approximation error into three components:

- **Cost mismatch** epsilon(s) = |h_M(s) - h_Mhat(s)|
- **Transition mismatch** Delta(s,a) on V_hat\* between the two models' dynamics
- **Propagated future error** through the true model's transitions

The bound 2\*alpha(s) >= V^{pi_hat\*}(s) - V\*(s) is compared against the sup-norm and weighted-norm bounds, showing significant tightness improvements in the operating region (sample-path / weighted-norm ratio ~ 0.83, sample-path / sup-norm ratio ~ 0.06).
