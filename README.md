# Weighted Approximation

Numerical experiments for weighted-norm and sample-path dependent bounds on suboptimality gaps arising from MDP model approximations.

## Overview

We study two test MDPs, each under a true model **M** and an approximate model **$\hat{M}$**:

- **Inventory MDP** -- models differ in demand distribution and holding cost; optimal policies are base-stock.
- **IoT remote transmission scheduling** (Exercise 5.4 of *Stochastic Control* notes at https://adityam.github.io/stochastic-control/mdps/intro.html#exercises, with centered binomial noise) -- models differ in noise drift and transmission cost; optimal policies are threshold bands.

The project computes and compares three types of bounds on the suboptimality gap $V^{\hat{\pi}^\star\} - V^\star$:

1. **Sup-norm bound** -- uniform constant bound over all states
2. **Weighted-norm bound** -- state-dependent bound using a Lyapunov weight function
3. **Sample-path bound** -- state-dependent bound via fixed-point iteration on local error certificates, tightest in the operating region

## Folder Structure

```
.
├── models.py                  # InventoryMDP and IoTModel classes (parameters, noise PMFs, per-step costs)
├── solvers.py                 # Value iteration (H-array trick), Bellman operators, policy evaluation
├── bounds.py                  # Kappa computation, weighted-norm bounds, sample-path fixed-point bound
├── plots.py                   # Plotting utilities for zoomed-in/zoomed-out figures
├── tests.py                   # Sanity checks for models, solvers, and bounds
├── data/                      # Generated data artifacts (e.g. policies_fh.csv)
├── docs/
│   ├── tex/                   # LaTeX derivations and computation-step sources
│   ├── pdf/                   # Rendered derivations and papers
│   └── notes/                 # Task specs and working notes
├── notebooks/
│   ├── Sample_Path_Bounds.ipynb     # Sample-path bound derivation and comparison (inventory)
│   ├── Finite_Horizon_Bounds.ipynb  # Finite-horizon and signed sample-path bounds (inventory)
│   ├── IoT_Bounds.ipynb             # Full bound suite on the IoT transmission-scheduling model
│   └── Paper_Figures.ipynb          # Figure reproduction
└── figures/
    ├── fig1-fig5/             # Figures from the model approximation paper
    ├── comparison/            # Sample-path vs existing bound comparison plots
    ├── finite_horizon/        # Finite-horizon and signed-bound figures
    ├── iot/                   # IoT model bound comparison figures
    └── misc/                  # Miscellaneous figure exports
```

## Computation Steps

`docs/pdf/computation_steps.pdf` contains the full set of steps for computing the sample dependent bound.

## Sample-Path Bounds Notebook

`notebooks/Sample_Path_Bounds.ipynb` implements the sample-path dependent AIS bound (Theorem 2, MDP Corollary). It computes a state-dependent function alpha(s) via fixed-point iteration that decomposes the approximation error into three components:

- **Cost mismatch** $\epsilon(s) = |h_M(s) - h_{\hat{M}}(s)|$
- **Transition mismatch** Delta(s,a) on $\hat{V}^\star$ between the two models' dynamics
- **Propagated future error** through the true model's transitions

The bound $2\alpha(s) >= V^{\hat{\pi}\star}(s) - V^\star(s)$ is compared against the sup-norm and weighted-norm bounds, showing significant tightness improvements in the operating region (sample-path / weighted-norm ratio ~ 0.83, sample-path / sup-norm ratio ~ 0.06).

## IoT Bounds Notebook

`notebooks/IoT_Bounds.ipynb` runs the same bound suite on the remote transmission-scheduling model: sync error $S_{t+1} = \mathrm{clip}(S_t + W_t)$ if idle, $\mathrm{clip}(W_t)$ if transmitting, with cost $\lambda a + (1-a)s^2$ and centered binomial noise $W = D - n/2$, $D \sim \mathrm{Bin}(10, q)$. True model $(q=0.4, \lambda=100)$ vs approximate $(q=0.5, \lambda=95)$: a drift mismatch plus an **action-dependent** cost mismatch $\epsilon(s,a) = 5a$. The restricted action set exploits the threshold-band policy structure (band contains 0 and lies within $\pm\lceil\sqrt{\lambda}\rceil$). The notebook validates truncation, bound validity, orderings, and the bit-exact collapse of the Theorem-3 lower bound ($L \equiv 0$); `docs/tex/experiments_section.tex` is the paper-style writeup: it restates the POMDP bounds (absolute-value and signed), states their MDP corollaries, and covers both experiments (inventory and IoT) with the absolute-value and signed bound computations referencing those corollaries.
