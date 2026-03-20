import numpy as np


def value_iteration(model, thres=1e-4):
    """Fast value iteration using the H-array trick (with dp fix for a=0).

    Returns (V_star, pi_star).
    """
    s_max = model.s_max
    W = model.W
    gamma = model.gamma
    p = model.p
    ch = model.ch
    cs = model.cs

    V = np.zeros(2 * s_max + 1)
    H = np.zeros(3 * s_max + 1)
    pi = np.zeros(len(V), dtype=int)

    while True:
        for z in range(-s_max, 2 * s_max + 1):
            H[z + s_max] = 0
            for w in range(len(W)):
                next_state = min(max(z - w, -s_max), s_max)
                H[z + s_max] += V[next_state + s_max] * W[w]

        delta = 0
        for s in range(-s_max, s_max + 1):
            h_s = model.h(s)
            # a=0 case: includes h(s) cost (the dp fix)
            opt = 0
            val = h_s + gamma * H[s + s_max]
            for a in range(1, s_max + 1):
                new_val = p * a + h_s + gamma * H[min(s + a, 2 * s_max) + s_max]
                if new_val <= val:
                    opt = a
                    val = new_val
            delta = max(delta, abs(val - V[s + s_max]))
            pi[s + s_max] = opt
            V[s + s_max] = val

        if delta < thres:
            break

    return V, pi


def bellman_opt_step(V, model):
    """One-step optimality Bellman operator (naive, for bound computation)."""
    s_max = model.s_max
    W = model.W
    gamma = model.gamma
    p = model.p
    num_states = model.num_states
    states = model.states

    Q = np.zeros((num_states, s_max + 1))
    for i in range(num_states):
        s = states[i]
        for a in range(s_max + 1):
            total = 0
            for w in range(len(W)):
                next_state = min(max(s + a - w, -s_max), s_max)
                total += V[int(next_state) + s_max] * W[w]
            Q[i, a] = p * a + model.h(s) + gamma * total

    pi = np.argmin(Q, axis=1)
    BV = Q[np.arange(len(Q)), pi]
    return BV


def bellman_pi_step(V, pi, model):
    """One-step Bellman operator for a given policy."""
    s_max = model.s_max
    W = model.W
    gamma = model.gamma
    p = model.p
    num_states = model.num_states
    states = model.states

    BV = np.zeros(num_states)
    for i in range(num_states):
        s = states[i]
        total = 0
        for w in range(len(W)):
            next_state = min(max(s + pi[i] - w, -s_max), s_max)
            total += V[int(next_state) + s_max] * W[w]
        BV[i] = p * pi[i] + model.h(s) + gamma * total
    return BV


def bellman_opt_alpha_beta_step(V, model, alpha=1, beta=0):
    """One-step optimality Bellman operator with (alpha, beta) cost transformation."""
    s_max = model.s_max
    W = model.W
    gamma = model.gamma
    p = model.p
    num_states = model.num_states
    states = model.states

    Q = np.zeros((num_states, s_max + 1))
    for i in range(num_states):
        s = states[i]
        for a in range(s_max + 1):
            total = 0
            for w in range(len(W)):
                next_state = min(max(s + a - w, -s_max), s_max)
                total += V[int(next_state) + s_max] * W[w]
            Q[i, a] = beta + alpha * (p * a + model.h(s)) + gamma * total

    pi = np.argmin(Q, axis=1)
    BV = Q[np.arange(len(Q)), pi]
    return BV


def bellman_pi_alpha_beta_step(V, pi, model, alpha=1, beta=0):
    """One-step Bellman operator for a given policy with (alpha, beta) cost transformation."""
    s_max = model.s_max
    W = model.W
    gamma = model.gamma
    p = model.p
    num_states = model.num_states
    states = model.states

    BV = np.zeros(num_states)
    for i in range(num_states):
        s = states[i]
        total = 0
        for w in range(len(W)):
            next_state = min(max(s + pi[i] - w, -s_max), s_max)
            total += V[int(next_state) + s_max] * W[w]
        BV[i] = beta + alpha * (p * pi[i] + model.h(s)) + gamma * total
    return BV


def policy_evaluation(model, pi, thres=1e-6):
    """Evaluate a policy on a model by iterating bellman_pi_step until convergence."""
    V = np.zeros(model.num_states)
    while True:
        V_new = bellman_pi_step(V, pi, model)
        delta = np.max(np.abs(V - V_new))
        V = V_new
        if delta < thres:
            break
    return V
