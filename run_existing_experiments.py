"""Generate all 10 paper figures for approximation_final.tex."""

import os
import shutil
import numpy as np
from models import InventoryMDP
from solvers import value_iteration, policy_evaluation
from bounds import (weight_function, compute_kappa, compute_kappa_all,
                    weighted_norm_bound_alpha_beta)
from plots import (plot_zoomed_out, plot_zoomed_in,
                   plot_single_bound_zoomed_out, plot_single_bound_zoomed_in,
                   plot_alpha_beta_zoomed_out, plot_alpha_beta_zoomed_in)


# ---------------------------------------------------------------------------
# Common setup
# ---------------------------------------------------------------------------

def setup():
    """Build models, run VI and policy evaluation (done once)."""
    M = InventoryMDP(s_max=500, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    M_hat = InventoryMDP(s_max=500, gamma=0.75, n=10, q=0.5, ch=3.8, cs=2, p=5)

    print("Running value iteration on true model M ...")
    V_star, pi_star = value_iteration(M)
    print("Running value iteration on approximate model M_hat ...")
    V_hat_star, pi_hat_star = value_iteration(M_hat)

    sigma_star = int(M.states[np.where(pi_star == 0)[0][0]])
    sigma_hat = int(M_hat.states[np.where(pi_hat_star == 0)[0][0]])
    print(f"Base-stock level sigma* = {sigma_star}")
    print(f"Base-stock level sigma_hat* = {sigma_hat}")

    print("Running policy evaluation ...")
    V_pi_hat_star = policy_evaluation(M, pi_hat_star)

    gap = V_pi_hat_star - V_star
    print(f"max(V_pi_hat_star - V_star) = {np.max(gap):.4f}")
    print(f"min(V_pi_hat_star - V_star) = {np.min(gap):.6f}")

    return M, M_hat, V_star, pi_star, V_hat_star, pi_hat_star, V_pi_hat_star


# ---------------------------------------------------------------------------
# Set 1: Weighted vs Sup-norm (Section 6.1, Fig 1-2) — 4 files
# ---------------------------------------------------------------------------

def generate_weighted_vs_sup(M, M_hat, V_hat_star, pi_star, pi_hat_star,
                              V_pi_hat_star):
    """Generate weight_zoomed_out/in.pdf and sup_zoomed_out/in.pdf."""
    print("\n=== Set 1: Weighted vs Sup-norm ===")
    states = M.states

    for ell, name in [(1.5e-2, "weight"), (0, "sup")]:
        weight = weight_function(M_hat, ell)
        kappa = max(
            compute_kappa(M, weight, pi_star),
            compute_kappa(M, weight, pi_hat_star),
            compute_kappa(M_hat, weight, pi_hat_star),
        )
        print(f"  {name}: ell={ell:.2e}, kappa={kappa:.6f}")

        bound = weighted_norm_bound_alpha_beta(
            V_hat_star, pi_hat_star, weight, M, M_hat, kappa,
            alpha=1, beta=0,
        )
        bound_curve = bound * weight
        print(f"  {name}: bound={bound:.4f}")

        label = r'$\ell = 0$' if ell == 0 else f'$\\ell = {ell:.1e}$'

        plot_single_bound_zoomed_out(
            states, V_pi_hat_star, bound_curve, label,
            filename=f"Figures/{name}_zoomed_out.pdf")
        plot_single_bound_zoomed_in(
            states, V_pi_hat_star, bound_curve, label,
            filename=f"Figures/{name}_zoomed_in.pdf")

    print("  Saved: weight_zoomed_out/in.pdf, sup_zoomed_out/in.pdf")


# ---------------------------------------------------------------------------
# Set 2: Multiple Weight Functions (Section 6.2, Fig 3) — 2 files
# ---------------------------------------------------------------------------

def generate_multi_weight(M, M_hat, V_hat_star, pi_star, pi_hat_star,
                           V_pi_hat_star):
    """Generate multi_out_500_new.pdf and multi_in_500_new.pdf."""
    print("\n=== Set 2: Multiple Weight Functions (policy stability) ===")
    states = M.states
    l_arr = np.linspace(0, 2.5e-2, 6)

    curves = []
    labels = []
    for ell in l_arr:
        weight = weight_function(M_hat, ell)
        kappa = max(
            compute_kappa(M, weight, pi_star),
            compute_kappa(M, weight, pi_hat_star),
            compute_kappa(M_hat, weight, pi_hat_star),
        )
        print(f"  ell={ell:.4e}  kappa={kappa:.6f}")
        assert kappa < 1 / M.gamma, f"kappa={kappa:.4f} >= 1/gamma"

        bound = weighted_norm_bound_alpha_beta(
            V_hat_star, pi_hat_star, weight, M, M_hat, kappa,
            alpha=1, beta=0,
        )
        curves.append(bound * weight)

        if ell == 0:
            labels.append(r'$\ell = 0$')
        else:
            labels.append(f'$\\ell = {ell:.2e}$')

    plot_zoomed_out(states, V_pi_hat_star, curves, labels,
                    filename="Figures/multi_out_500_new.pdf")
    plot_zoomed_in(states, V_pi_hat_star, curves, labels,
                   filename="Figures/multi_in_500_new.pdf")
    print("  Saved: multi_out_500_new.pdf, multi_in_500_new.pdf")


# ---------------------------------------------------------------------------
# Set 3: Alpha-Beta Bounds (Section 6.3, Fig 4) — 2 files
# ---------------------------------------------------------------------------

def generate_alpha_beta(M, M_hat, V_hat_star, pi_star, pi_hat_star,
                         V_pi_hat_star):
    """Generate alpha_beta_bound_out.pdf and alpha_beta_bound_in.pdf."""
    print("\n=== Set 3: Alpha-Beta Bounds ===")
    states = M.states
    ell = 1.5e-2
    weight = weight_function(M_hat, ell)

    kappa = max(
        compute_kappa(M, weight, pi_star),
        compute_kappa(M, weight, pi_hat_star),
        compute_kappa(M_hat, weight, pi_hat_star),
    )
    print(f"  ell={ell:.2e}, kappa={kappa:.6f}")

    configs = [
        (1, 0, r'$\alpha=1, \beta=0$'),
        (0.98, 0.8, r'$\alpha=0.98, \beta=0.8$'),
    ]

    curves = []
    labels = []
    for alpha, beta, label in configs:
        bound = weighted_norm_bound_alpha_beta(
            V_hat_star, pi_hat_star, weight, M, M_hat, kappa,
            alpha=alpha, beta=beta,
        )
        curves.append(bound * weight)
        labels.append(label)
        print(f"  alpha={alpha}, beta={beta}: bound={bound:.4f}")

    plot_alpha_beta_zoomed_out(states, V_pi_hat_star, curves, labels,
                               filename="Figures/alpha_beta_bound_out.pdf")
    plot_alpha_beta_zoomed_in(states, V_pi_hat_star, curves, labels,
                              filename="Figures/alpha_beta_bound_in.pdf")
    print("  Saved: alpha_beta_bound_out.pdf, alpha_beta_bound_in.pdf")


# ---------------------------------------------------------------------------
# Set 4: Model Stability (Section 5, Fig 5) — 2 files
# ---------------------------------------------------------------------------

def generate_model_stability(M, M_hat, V_hat_star, pi_hat_star,
                              V_pi_hat_star):
    """Generate multi_out_500_act_new.pdf and multi_in_500_act_new.pdf."""
    print("\n=== Set 4: Model Stability (kappa_all) ===")
    states = M.states
    l_arr = np.linspace(0, 2.5e-4, 6)

    curves = []
    labels = []
    for ell in l_arr:
        weight = weight_function(M_hat, ell)
        kappa = max(
            compute_kappa_all(M, weight),
            compute_kappa_all(M_hat, weight),
        )
        if kappa >= 1 / M.gamma:
            print(f"  ell={ell:.4e}  kappa={kappa:.6f}  SKIPPED (>= 1/gamma)")
            continue
        print(f"  ell={ell:.4e}  kappa={kappa:.6f}")

        bound = weighted_norm_bound_alpha_beta(
            V_hat_star, pi_hat_star, weight, M, M_hat, kappa,
            alpha=1, beta=0,
        )
        curves.append(bound * weight)

        if ell == 0:
            labels.append(r'$\ell = 0$')
        else:
            labels.append(f'$\\ell = {ell:.2e}$')

    plot_zoomed_out(states, V_pi_hat_star, curves, labels,
                    filename="Figures/multi_out_500_act_new.pdf")
    plot_zoomed_in(states, V_pi_hat_star, curves, labels,
                   filename="Figures/multi_in_500_act_new.pdf")
    print("  Saved: multi_out_500_act_new.pdf, multi_in_500_act_new.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    M, M_hat, V_star, pi_star, V_hat_star, pi_hat_star, V_pi_hat_star = setup()

    generate_weighted_vs_sup(M, M_hat, V_hat_star, pi_star, pi_hat_star,
                              V_pi_hat_star)
    generate_multi_weight(M, M_hat, V_hat_star, pi_star, pi_hat_star,
                           V_pi_hat_star)
    generate_alpha_beta(M, M_hat, V_hat_star, pi_star, pi_hat_star,
                         V_pi_hat_star)
    generate_model_stability(M, M_hat, V_hat_star, pi_hat_star,
                              V_pi_hat_star)

    # Copy into fig1–fig5 subfolders for easy comparison with the paper
    fig_map = {
        "fig1": ["weight_zoomed_out.pdf", "sup_zoomed_out.pdf"],
        "fig2": ["weight_zoomed_in.pdf", "sup_zoomed_in.pdf"],
        "fig3": ["multi_out_500_new.pdf", "multi_in_500_new.pdf"],
        "fig4": ["alpha_beta_bound_out.pdf", "alpha_beta_bound_in.pdf"],
        "fig5": ["multi_out_500_act_new.pdf", "multi_in_500_act_new.pdf"],
    }
    for folder, files in fig_map.items():
        dest = os.path.join("Figures", folder)
        os.makedirs(dest, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join("Figures", f), dest)

    print("\nDone — all 10 figures saved to Figures/ and Figures/fig1–fig5/")


if __name__ == "__main__":
    main()
