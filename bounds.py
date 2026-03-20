import numpy as np
from solvers import bellman_opt_step, bellman_pi_step
from solvers import bellman_opt_alpha_beta_step, bellman_pi_alpha_beta_step


def weight_function(model, ell):
    """Compute weight vector w(s) = 1 + ell * h_hat(s)."""
    return 1 + ell * model.h_vec(model.states)


def compute_kappa(model, weight, pi):
    """Compute kappa for a specific policy: max_s E[w(s')] / w(s)."""
    s_max = model.s_max
    W = model.W
    num_states = model.num_states
    states = model.states

    LHS = np.zeros(num_states)
    for i, s in enumerate(states):
        for w in range(len(W)):
            next_state = min(max(s + pi[i] - w, -s_max), s_max)
            LHS[i] += weight[int(next_state) + s_max] * W[w]
    return np.max(LHS / weight)


def compute_kappa_all(model, weight):
    """Compute kappa over all actions: max_{s,a} E[w(s')] / w(s)."""
    s_max = model.s_max
    W = model.W
    num_states = model.num_states
    num_actions = model.num_actions
    states = model.states

    LHS = np.zeros((num_states, num_actions))
    for i in range(num_states):
        for j in range(num_actions):
            for w in range(len(W)):
                next_state = min(max(states[i] + j - w, -s_max), s_max)
                LHS[i, j] += weight[int(next_state) + s_max] * W[w]
    return np.max(LHS / weight[:, None])


def weighted_norm_bound(V_hat_star, pi_hat_star, weight, M, M_hat, kappa):
    """First bound of Thm. 1 (bound_v_pi)."""
    gamma = M.gamma
    mismatch_opt = np.max(
        np.abs(bellman_opt_step(V_hat_star, M) - bellman_opt_step(V_hat_star, M_hat))
        / weight
    )
    mismatch_pi = np.max(
        np.abs(bellman_pi_step(V_hat_star, pi_hat_star, M)
               - bellman_pi_step(V_hat_star, pi_hat_star, M_hat))
        / weight
    )
    return (1 / (1 - gamma * kappa)) * (mismatch_opt + mismatch_pi)


def weighted_norm_bound_alpha_beta(V_hat_star, pi_hat_star, weight,
                                    M, M_hat, kappa, alpha=1, beta=0):
    """Bound with (alpha, beta) cost transformation (bound_alpha_beta)."""
    gamma = M.gamma
    mismatch_opt = np.max(
        np.abs(bellman_opt_alpha_beta_step(V_hat_star, M, alpha, beta)
               - bellman_opt_step(V_hat_star, M_hat))
        / weight
    )
    mismatch_pi = np.max(
        np.abs(bellman_pi_alpha_beta_step(V_hat_star, pi_hat_star, M, alpha, beta)
               - bellman_pi_step(V_hat_star, pi_hat_star, M_hat))
        / weight
    )
    return (1 / (1 - gamma * kappa)) * (1 / alpha) * (mismatch_opt + mismatch_pi)


def sample_path_bound(M, M_hat, V_hat_star, pi_star, pi_hat_star,
                      tol=1e-10, max_iter=1000):
    """Sample-path dependent AIS bound (Theorem 2 / MDP Corollary).

    Returns (alpha_max, alpha_sup, convergence_history).
    alpha_max uses only the two relevant actions pi_star(s), pi_hat_star(s).
    alpha_sup maximises over all actions.
    The actual suboptimality bound is 2*alpha(s).
    """
    s_max = M.s_max
    gamma = M.gamma
    W_M = M.W          # true demand PMF
    W_Mh = M_hat.W     # approx demand PMF
    n_demand = len(W_M)  # n+1

    num_states = M.num_states  # 2*s_max + 1
    H_len = 3 * s_max + 1     # indices z in [-s_max, 2*s_max]

    # --- 1a. Cost mismatch epsilon(s) = |h_M(s) - h_Mhat(s)| ---
    states = M.states
    h_M = M.h_vec(states)
    h_Mh = M_hat.h_vec(states)
    epsilon = np.abs(h_M - h_Mh)  # shape (num_states,)

    # --- 1b. Transition mismatch H_delta[z] = |H_M[z] - H_Mhat[z]| on V_hat_star ---
    H_M = np.zeros(H_len)
    H_Mh = np.zeros(H_len)
    for z in range(-s_max, 2 * s_max + 1):
        for w in range(n_demand):
            ns = min(max(z - w, -s_max), s_max)
            H_M[z + s_max] += W_M[w] * V_hat_star[ns + s_max]
            H_Mh[z + s_max] += W_Mh[w] * V_hat_star[ns + s_max]
    H_delta = np.abs(H_M - H_Mh)  # shape (H_len,)

    # --- 1c. Fixed-point iteration for alpha ---
    alpha = np.zeros(num_states)
    history = []

    for iteration in range(max_iter):
        # Build H_alpha[z] = sum_w W_M[w] * alpha(clip(z-w))
        H_alpha = np.zeros(H_len)
        for z in range(-s_max, 2 * s_max + 1):
            for w in range(n_demand):
                ns = min(max(z - w, -s_max), s_max)
                H_alpha[z + s_max] += W_M[w] * alpha[ns + s_max]

        # alpha_max: max over the two relevant actions
        alpha_new_max = np.zeros(num_states)
        for i, s in enumerate(states):
            s_int = int(s)
            best = -np.inf
            for a in [pi_star[i], pi_hat_star[i]]:
                z_idx = min(s_int + a, 2 * s_max) + s_max
                beta_sa = epsilon[i] + gamma * H_alpha[z_idx] + H_delta[z_idx]
                if beta_sa > best:
                    best = beta_sa
            alpha_new_max[i] = best

        # alpha_sup: max over all actions
        # For each s, a in 0..s_max: z = s+a clipped to 2*s_max
        # beta(s,a) = eps(s) + gamma*H_alpha[z] + H_delta[z]
        # Since eps(s) is const w.r.t. a: alpha_sup(s) = eps(s) + max_a[gamma*H_alpha[z]+H_delta[z]]
        alpha_new_sup = np.zeros(num_states)
        combo = gamma * H_alpha + H_delta  # shape (H_len,)
        for i, s in enumerate(states):
            s_int = int(s)
            # a ranges 0..s_max, so z = s_int+a ranges s_int..s_int+s_max
            z_lo = s_int + s_max       # z_idx for a=0
            z_hi = min(s_int + s_max, 2 * s_max) + s_max  # z_idx for a=s_max
            alpha_new_sup[i] = epsilon[i] + np.max(combo[z_lo:z_hi + 1])

        diff = np.max(np.abs(alpha_new_max - alpha))
        history.append(diff)

        alpha = alpha_new_max  # iterate on alpha_max

        if diff < tol:
            break

    # Run alpha_sup to convergence separately (using its own fixed-point)
    alpha_sup = np.zeros(num_states)
    for iteration in range(max_iter):
        H_alpha_sup = np.zeros(H_len)
        for z in range(-s_max, 2 * s_max + 1):
            for w in range(n_demand):
                ns = min(max(z - w, -s_max), s_max)
                H_alpha_sup[z + s_max] += W_M[w] * alpha_sup[ns + s_max]

        combo_sup = gamma * H_alpha_sup + H_delta
        alpha_new_sup = np.zeros(num_states)
        for i, s in enumerate(states):
            s_int = int(s)
            z_lo = s_int + s_max
            z_hi = min(s_int + s_max, 2 * s_max) + s_max
            alpha_new_sup[i] = epsilon[i] + np.max(combo_sup[z_lo:z_hi + 1])

        diff_sup = np.max(np.abs(alpha_new_sup - alpha_sup))
        alpha_sup = alpha_new_sup
        if diff_sup < tol:
            break

    return alpha, alpha_sup, history
