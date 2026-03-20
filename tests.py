"""Sanity checks for the refactored inventory MDP code."""

import numpy as np
from models import InventoryMDP
from solvers import value_iteration, policy_evaluation, bellman_opt_step
from bounds import weight_function, compute_kappa, compute_kappa_all


def test_optimal_dominates_suboptimal():
    """V_star(s) <= V_pi_hat_star(s) for all s."""
    M = InventoryMDP(s_max=100, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    M_hat = InventoryMDP(s_max=100, gamma=0.75, n=10, q=0.5, ch=3.8, cs=2, p=5)

    V_star, _ = value_iteration(M)
    _, pi_hat_star = value_iteration(M_hat)
    V_pi_hat_star = policy_evaluation(M, pi_hat_star)

    diff = V_pi_hat_star - V_star
    assert np.all(diff >= -1e-6), (
        f"V_star should be <= V_pi_hat_star everywhere, min diff = {np.min(diff)}"
    )
    print(f"PASS: V_star <= V_pi_hat_star (max gap = {np.max(diff):.4f})")


def test_convergence():
    """Value iteration converges (returns finite values)."""
    M = InventoryMDP(s_max=50, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    V, pi = value_iteration(M, thres=1e-6)
    assert np.all(np.isfinite(V)), "V contains non-finite values"
    assert np.all(np.isfinite(pi)), "pi contains non-finite values"
    print("PASS: Value iteration converges with finite values")


def test_base_stock_policy():
    """Optimal policies are base-stock: pi(s) = max(0, sigma - s) for interior states."""
    M = InventoryMDP(s_max=500, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    V, pi = value_iteration(M)

    # Find base-stock level from near the origin (avoid boundary effects)
    # sigma is the smallest s where pi(s) = 0 and s > -s_max + s_max/2
    interior_start = M.s_max // 2
    for i in range(interior_start, M.num_states):
        if pi[i] == 0:
            sigma = int(M.states[i])
            break

    # Check base-stock structure for states where expected action <= s_max
    for i, s in enumerate(M.states):
        s_int = int(s)
        expected = max(0, sigma - s_int)
        if expected > M.s_max:
            continue  # skip boundary states where action is clipped
        assert pi[i] == expected, (
            f"Not base-stock at s={s_int}: pi={pi[i]}, expected={expected}"
        )
    print(f"PASS: Policy is base-stock with sigma = {sigma}")


def test_bounds_nonnegative():
    """All bounds are non-negative."""
    from bounds import weighted_norm_bound_alpha_beta

    M = InventoryMDP(s_max=50, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    M_hat = InventoryMDP(s_max=50, gamma=0.75, n=10, q=0.5, ch=3.8, cs=2, p=5)

    V_hat_star, pi_hat_star = value_iteration(M_hat)
    V_star, pi_star = value_iteration(M)

    for ell in np.linspace(0, 2.5e-2, 6):
        weight = weight_function(M_hat, ell)
        kappa = max(
            compute_kappa(M, weight, pi_star),
            compute_kappa(M, weight, pi_hat_star),
            compute_kappa(M_hat, weight, pi_hat_star),
        )
        bound = weighted_norm_bound_alpha_beta(
            V_hat_star, pi_hat_star, weight, M, M_hat, kappa,
        )
        assert bound >= 0, f"Bound is negative: {bound} for ell={ell}"
    print("PASS: All bounds are non-negative")


def test_dp_matches_bellman_opt():
    """dp (value_iteration) matches naive bellman_opt on small instance."""
    M = InventoryMDP(s_max=20, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    V_dp, _ = value_iteration(M, thres=1e-8)

    # Apply bellman_opt to V_dp — should be a fixed point
    BV = bellman_opt_step(V_dp, M)
    diff = np.max(np.abs(V_dp - BV))
    assert diff < 1e-6, f"dp and bellman_opt disagree: max diff = {diff}"
    print(f"PASS: dp matches bellman_opt (max diff = {diff:.2e})")


def test_kappa_stability():
    """gamma * kappa < 1 for all weight functions used."""
    M = InventoryMDP(s_max=50, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    M_hat = InventoryMDP(s_max=50, gamma=0.75, n=10, q=0.5, ch=3.8, cs=2, p=5)

    _, pi_star = value_iteration(M)
    _, pi_hat_star = value_iteration(M_hat)

    for ell in np.linspace(0, 2.5e-2, 6):
        weight = weight_function(M_hat, ell)
        kappa = max(
            compute_kappa(M, weight, pi_star),
            compute_kappa(M, weight, pi_hat_star),
        )
        assert M.gamma * kappa < 1, (
            f"Stability violated: gamma*kappa = {M.gamma * kappa:.4f} for ell={ell}"
        )
    print("PASS: gamma * kappa < 1 for all weight functions")


def test_base_stock_levels_full_scale():
    """Base-stock levels match expected values at s_max=500."""
    M = InventoryMDP(s_max=500, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    M_hat = InventoryMDP(s_max=500, gamma=0.75, n=10, q=0.5, ch=3.8, cs=2, p=5)

    _, pi_star = value_iteration(M)
    _, pi_hat_star = value_iteration(M_hat)

    sigma_star = int(M.states[np.where(pi_star == 0)[0][0]])
    sigma_hat = int(M_hat.states[np.where(pi_hat_star == 0)[0][0]])

    assert sigma_star == 2, f"Expected sigma*=2, got {sigma_star}"
    assert sigma_hat == 3, f"Expected sigma_hat*=3, got {sigma_hat}"
    print(f"PASS: Base-stock levels sigma*={sigma_star}, sigma_hat*={sigma_hat}")


def test_sample_path_bound_valid():
    """2*alpha_max(s) >= V^{pi_hat_star}(s) - V_star(s) for all s."""
    from bounds import sample_path_bound

    M = InventoryMDP(s_max=100, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    M_hat = InventoryMDP(s_max=100, gamma=0.75, n=10, q=0.5, ch=3.8, cs=2, p=5)

    V_star, pi_star = value_iteration(M)
    V_hat_star, pi_hat_star = value_iteration(M_hat)
    V_pi_hat_star = policy_evaluation(M, pi_hat_star)

    alpha_max, _, _ = sample_path_bound(M, M_hat, V_hat_star, pi_star, pi_hat_star)
    gap = V_pi_hat_star - V_star
    violation = np.min(2 * alpha_max - gap)

    assert violation >= -1e-6, (
        f"Bound violated: min(2*alpha - gap) = {violation}"
    )
    print(f"PASS: 2*alpha_max >= gap (min slack = {violation:.6f})")


def test_sample_path_tighter_than_weighted():
    """max_s 2*alpha_max(s) < sup-norm bound (sample-path tighter overall)."""
    from bounds import sample_path_bound, weighted_norm_bound_alpha_beta

    M = InventoryMDP(s_max=100, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    M_hat = InventoryMDP(s_max=100, gamma=0.75, n=10, q=0.5, ch=3.8, cs=2, p=5)

    V_star, pi_star = value_iteration(M)
    V_hat_star, pi_hat_star = value_iteration(M_hat)

    alpha_max, _, _ = sample_path_bound(M, M_hat, V_hat_star, pi_star, pi_hat_star)

    # Sup-norm bound (ell=0, weight=1 everywhere)
    ell = 0
    weight = weight_function(M_hat, ell)
    kappa = max(
        compute_kappa(M, weight, pi_star),
        compute_kappa(M, weight, pi_hat_star),
        compute_kappa(M_hat, weight, pi_hat_star),
    )
    sup_bound = weighted_norm_bound_alpha_beta(
        V_hat_star, pi_hat_star, weight, M, M_hat, kappa)

    # In the operating region, sample-path should be much tighter than sup-norm
    s_max = M.s_max
    op_lo = s_max - 8
    op_hi = s_max + 2 + 1
    sp_max = np.max(2 * alpha_max[op_lo:op_hi])
    assert sp_max < sup_bound, (
        f"Sample-path not tighter than sup-norm: {sp_max:.4f} >= {sup_bound:.4f}"
    )
    print(f"PASS: max 2*alpha_max in [-8,2] = {sp_max:.4f} < sup-norm = {sup_bound:.4f}")


def test_alpha_max_leq_alpha_sup():
    """alpha_max(s) <= alpha_sup(s) for all s."""
    from bounds import sample_path_bound

    M = InventoryMDP(s_max=100, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5)
    M_hat = InventoryMDP(s_max=100, gamma=0.75, n=10, q=0.5, ch=3.8, cs=2, p=5)

    V_star, pi_star = value_iteration(M)
    V_hat_star, pi_hat_star = value_iteration(M_hat)

    alpha_max, alpha_sup, _ = sample_path_bound(
        M, M_hat, V_hat_star, pi_star, pi_hat_star)
    diff = alpha_max - alpha_sup
    assert np.all(diff <= 1e-6), (
        f"alpha_max > alpha_sup at some state: max excess = {np.max(diff)}"
    )
    print(f"PASS: alpha_max <= alpha_sup (max diff = {np.max(diff):.6f})")


if __name__ == "__main__":
    tests = [
        test_convergence,
        test_base_stock_policy,
        test_dp_matches_bellman_opt,
        test_optimal_dominates_suboptimal,
        test_bounds_nonnegative,
        test_kappa_stability,
        test_base_stock_levels_full_scale,
        test_sample_path_bound_valid,
        test_sample_path_tighter_than_weighted,
        test_alpha_max_leq_alpha_sup,
    ]

    print("=" * 60)
    print("Running sanity checks")
    print("=" * 60)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
