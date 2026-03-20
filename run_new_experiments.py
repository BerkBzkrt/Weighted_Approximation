"""Phase 3: Sample-path dependent AIS bounds — comparison experiments."""

import numpy as np
from models import InventoryMDP
from solvers import value_iteration, policy_evaluation
from bounds import (weight_function, compute_kappa,
                    weighted_norm_bound_alpha_beta, sample_path_bound)
from plots import (plot_shaded_comparison, plot_bound_comparison,
                   plot_error_decomposition, plot_convergence,
                   plot_max_vs_sup)


def main():
    # ---- 1. Setup (same as existing pipeline) ----
    M = InventoryMDP(s_max=500, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    M_hat = InventoryMDP(s_max=500, gamma=0.75, n=10, q=0.5, ch=3.8, cs=2, p=5)

    print("Running value iteration on M ...")
    V_star, pi_star = value_iteration(M)
    print("Running value iteration on M_hat ...")
    V_hat_star, pi_hat_star = value_iteration(M_hat)

    sigma_star = int(M.states[np.where(pi_star == 0)[0][0]])
    sigma_hat = int(M_hat.states[np.where(pi_hat_star == 0)[0][0]])
    print(f"Base-stock levels: sigma*={sigma_star}, sigma_hat*={sigma_hat}")

    print("Running policy evaluation ...")
    V_pi_hat_star = policy_evaluation(M, pi_hat_star)
    gap = V_pi_hat_star - V_star
    print(f"Max suboptimality gap: {np.max(gap):.4f}")

    # ---- 2. Compute sample-path bound ----
    print("\nComputing sample-path bound (fixed-point iteration) ...")
    alpha_max, alpha_sup, history = sample_path_bound(
        M, M_hat, V_hat_star, pi_star, pi_hat_star)
    print(f"  Converged in {len(history)} iterations")
    print(f"  max 2*alpha_max = {2 * np.max(alpha_max):.4f}")
    print(f"  max 2*alpha_sup = {2 * np.max(alpha_sup):.4f}")

    # ---- 3. Compute existing bounds for comparison ----
    states = M.states
    s_max = M.s_max

    # Sup-norm (ell=0, weight=1)
    ell_sup = 0
    w_sup = weight_function(M_hat, ell_sup)
    kappa_sup = max(
        compute_kappa(M, w_sup, pi_star),
        compute_kappa(M, w_sup, pi_hat_star),
        compute_kappa(M_hat, w_sup, pi_hat_star),
    )
    bound_sup_scalar = weighted_norm_bound_alpha_beta(
        V_hat_star, pi_hat_star, w_sup, M, M_hat, kappa_sup)
    sup_bound_curve = bound_sup_scalar * w_sup  # constant since w=1
    print(f"\n  Sup-norm bound: {bound_sup_scalar:.4f}")

    # Weighted-norm (ell=1.5e-2)
    ell_w = 1.5e-2
    w_w = weight_function(M_hat, ell_w)
    kappa_w = max(
        compute_kappa(M, w_w, pi_star),
        compute_kappa(M, w_w, pi_hat_star),
        compute_kappa(M_hat, w_w, pi_hat_star),
    )
    bound_w_scalar = weighted_norm_bound_alpha_beta(
        V_hat_star, pi_hat_star, w_w, M, M_hat, kappa_w)
    weighted_bound_curve = bound_w_scalar * w_w
    print(f"  Weighted-norm bound (ell={ell_w}): {bound_w_scalar:.4f}")

    # ---- 4. Generate comparison plots ----
    out_dir = "Figures/comparison"

    # Plot 1a: Shaded comparison — zoomed out
    plot_shaded_comparison(
        states, V_pi_hat_star, V_star,
        sup_bound_curve, weighted_bound_curve, 2 * alpha_max,
        filename=f"{out_dir}/shaded_comparison_zoomed_out.pdf")

    # Plot 1b: Shaded comparison — zoomed in
    # Restrict to [-10, 10] for zoomed version
    lo = s_max - 10
    hi = s_max + 10 + 1
    plot_shaded_comparison(
        states[lo:hi], V_pi_hat_star[lo:hi], V_star[lo:hi],
        sup_bound_curve[lo:hi], weighted_bound_curve[lo:hi],
        2 * alpha_max[lo:hi],
        filename=f"{out_dir}/shaded_comparison_zoomed_in.pdf")

    # Plot 2: Direct bound comparison (zoomed)
    plot_bound_comparison(
        states, gap, alpha_max, alpha_sup,
        weighted_bound_curve, sup_bound_curve,
        filename=f"{out_dir}/bound_comparison.pdf")

    # Plot 3: Error decomposition
    # Need epsilon, delta_at_pi_hat, propagated = gamma * E[alpha | s, pi_hat]
    epsilon = np.abs(M.h_vec(states) - M_hat.h_vec(states))

    # delta at pi_hat_star: H_delta[s + pi_hat(s)]
    W_M = M.W
    W_Mh = M_hat.W
    n_demand = len(W_M)
    H_len = 3 * s_max + 1
    H_M_v = np.zeros(H_len)
    H_Mh_v = np.zeros(H_len)
    for z in range(-s_max, 2 * s_max + 1):
        for w in range(n_demand):
            ns = min(max(z - w, -s_max), s_max)
            H_M_v[z + s_max] += W_M[w] * V_hat_star[ns + s_max]
            H_Mh_v[z + s_max] += W_Mh[w] * V_hat_star[ns + s_max]
    H_delta = np.abs(H_M_v - H_Mh_v)

    delta_at_pi_hat = np.zeros(M.num_states)
    for i, s in enumerate(states):
        z_idx = min(int(s) + pi_hat_star[i], 2 * s_max) + s_max
        delta_at_pi_hat[i] = H_delta[z_idx]

    # propagated = gamma * E[alpha_max(s') | s, pi_hat_star(s)]
    H_alpha = np.zeros(H_len)
    for z in range(-s_max, 2 * s_max + 1):
        for w in range(n_demand):
            ns = min(max(z - w, -s_max), s_max)
            H_alpha[z + s_max] += W_M[w] * alpha_max[ns + s_max]
    propagated = np.zeros(M.num_states)
    for i, s in enumerate(states):
        z_idx = min(int(s) + pi_hat_star[i], 2 * s_max) + s_max
        propagated[i] = M.gamma * H_alpha[z_idx]

    plot_error_decomposition(
        states, epsilon, delta_at_pi_hat, propagated,
        filename=f"{out_dir}/error_decomposition.pdf")

    # Plot 4: Convergence
    plot_convergence(history, filename=f"{out_dir}/convergence.pdf")

    # Plot 5: Max vs sup
    plot_max_vs_sup(states, alpha_max, alpha_sup,
                    filename=f"{out_dir}/max_vs_sup.pdf")

    print(f"\nAll 6 comparison PDFs saved to {out_dir}/")

    # ---- 5. Key metrics ----
    print("\n=== Key Metrics ===")
    # Operating region check
    op_lo = s_max - 8  # state -8
    op_hi = s_max + 2 + 1  # state +2 inclusive
    print(f"Operating region [-8, 2]:")
    print(f"  max gap:             {np.max(gap[op_lo:op_hi]):.4f}")
    print(f"  max 2*alpha_max:     {np.max(2*alpha_max[op_lo:op_hi]):.4f}")
    print(f"  max weighted bound:  {np.max(weighted_bound_curve[op_lo:op_hi]):.4f}")
    print(f"  sup bound:           {bound_sup_scalar:.4f}")

    # Validity check
    violation = np.min(2 * alpha_max - gap)
    print(f"\n  min(2*alpha_max - gap) = {violation:.6f} (should be >= 0)")
    print(f"  alpha_max <= alpha_sup everywhere: "
          f"{np.all(alpha_max <= alpha_sup + 1e-10)}")


if __name__ == "__main__":
    main()
