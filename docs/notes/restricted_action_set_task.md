# Restricted Action Set for the Sample-Path Bound

## Background

The sample-path bound computes $\alpha(s)$ via fixed-point iteration. At each iteration, the $\alpha$ update takes the form:

$$\alpha^{(k+1)}(s) = \max_{a \in A(s)} \beta^{(k)}(s, a)$$

where

$$\beta^{(k)}(s, a) = \varepsilon(s, a) + \gamma \sum_{s'} \alpha^{(k)}(s') P(s'|s,a) + \Delta(s, a)$$

There are currently two variants:

- **alpha_max**: $A(s) = \{\pi^\star(s), \hat\pi^\star(s)\}$ — requires knowing the true model's optimal policy $\pi^\star$.
- **alpha_sup**: $A(s) = \mathcal{A} = \{0, 1, \ldots, S_{\max}\}$ — does not require $\pi^\star$ but searches over all 501 actions.

## The Problem

The `alpha_max` variant uses $\pi^\star(s)$, the optimal policy of the true model $M$, which was computed by running value iteration on $M$. In practice, if we don't have full access to the true model (or don't want to solve it), we may not know $\pi^\star$. We need a middle ground: tighter than `alpha_sup` but not requiring $\pi^\star$.

## Solution: Restricted Action Set via Base-Stock Structure

For the inventory management MDP, we know from classical inventory theory that the optimal policy under both models is a **base-stock policy**:

$$\pi^\star(s) = \max(0, \sigma^\star - s)$$

for some base-stock level $\sigma^\star \in \{0, 1, \ldots, S_{\max}\}$. Similarly, $\hat\pi^\star(s) = \max(0, \hat\sigma - s)$ with $\hat\sigma = 3$.

We don't need to know $\sigma^\star$ exactly. We only need to know that it lies in some set $\Sigma$. The key structural fact is:

**The optimal base-stock level satisfies $\sigma^\star \leq n$**, where $n = 10$ is the maximum possible demand (the size parameter of the Binomial distribution). The reason: at any base-stock level $\sigma > n$, shortage is impossible since demand $D \leq n < \sigma$, so the only effect of increasing $\sigma$ further is higher holding and ordering costs. Reducing $\sigma$ to $n$ saves on both without introducing any shortage risk.

Combined with the trivial lower bound $\sigma^\star \geq 0$, we have:

$$\sigma^\star \in \Sigma = \{0, 1, 2, \ldots, n\} = \{0, 1, 2, \ldots, 10\}$$

## How This Restricts the Action Set

A base-stock policy with level $\sigma$ takes action:
- $a = \sigma - s$ if $s < \sigma$ (order up to $\sigma$)
- $a = 0$ if $s \geq \sigma$ (don't order)

So $a = \max(0, \sigma - s)$.

Since $\pi^\star(s) = \max(0, \sigma^\star - s)$ for some $\sigma^\star \in \Sigma$, the action $\pi^\star(s)$ must lie in:

$$\{\max(0, \sigma - s) : \sigma \in \Sigma\}$$

The restricted action set at each state is:

$$A(s) = \{\hat\pi^\star(s)\} \cup \{\max(0, \sigma - s) : \sigma \in \Sigma\}$$

This is guaranteed to contain both $\pi^\star(s)$ and $\hat\pi^\star(s)$ (since $\hat\sigma = 3 \in \Sigma$), so the Theorem 3 condition $\alpha(s) \geq \max\{\beta(s, \pi^\star(s)), \beta(s, \hat\pi^\star(s))\}$ is satisfied.

## Size of the Restricted Action Set

For each state $s$:

- If $s \geq n = 10$: all candidate base-stock actions give $\max(0, \sigma - s) = 0$ (since $\sigma \leq 10 \leq s$), plus $\hat\pi^\star(s) = 0$. So $|A(s)| = 1$.
- If $s = 0$: candidate actions are $\{0, 1, 2, \ldots, 10\}$ plus $\hat\pi^\star(0) = 3$ (already in the set). So $|A(s)| = 11$.
- In general: $|A(s)| \leq |\Sigma| + 1 = 12$.

Compare with $|\mathcal{A}| = 501$ for `alpha_sup`. The restricted set is at most 12 actions regardless of state.

## Implementation

### New variant: `alpha_restricted`

Add a third variant alongside the existing `alpha_max` and `alpha_sup`.

**Inputs (same as existing):**
- `M`, `M_hat`: the two inventory MDP models
- `V_hat_star`: optimal value function of $\hat{M}$
- `pi_hat_star`: optimal policy of $\hat{M}$
- Note: `pi_star` is **NOT** needed for this variant

**Additional input:**
- `Sigma`: the set of candidate base-stock levels, default `range(0, M.n + 1)` which gives $\{0, 1, \ldots, 10\}$

**Precomputation (same as existing):**
- `epsilon[s] = |h_M(s) - h_M_hat(s)|` for all $s$
- `H_delta[z] = |H_M[z] - H_M_hat[z]|` for all $z$ (transition mismatch via H-array)

**Fixed-point iteration:**

At each iteration $k$, after building `H_alpha[z]` as before:

```python
# For each state s, compute the restricted action set and take the max
alpha_new_restricted = np.zeros(num_states)
for i, s in enumerate(states):
    s_int = int(s)
    best = -np.inf
    
    # Candidate actions from base-stock structure
    # For each sigma in Sigma, the action is max(0, sigma - s)
    candidate_actions = set()
    candidate_actions.add(pi_hat_star[i])  # always include hat_pi_star
    for sigma in Sigma:
        candidate_actions.add(max(0, sigma - s_int))
    
    for a in candidate_actions:
        z_idx = min(s_int + a, 2 * s_max) + s_max
        beta_sa = epsilon[i] + gamma * H_alpha[z_idx] + H_delta[z_idx]
        if beta_sa > best:
            best = beta_sa
    alpha_new_restricted[i] = best
```

**With the H-array trick**, each candidate action $a$ at state $s$ maps to post-order level $z = s + a$. Since the candidate actions come from base-stock levels, $z = s + \max(0, \sigma - s) = \max(s, \sigma)$. So the candidate post-order levels are:

$$\{z : z = \max(s, \sigma), \sigma \in \Sigma\} \cup \{\max(s, \hat\sigma)\}$$

For $s \geq n$, all candidates collapse to $z = s$ (a single point). For $s < 0$, the candidates are $\{s\} \cup \Sigma$ (since $\max(0, \sigma - s) = \sigma - s > 0$ for all $\sigma \geq 0$ when $s < 0$, giving $z = \sigma$... actually let me be more careful).

For $s < 0$: $a = \max(0, \sigma - s) = \sigma - s$ (since $\sigma \geq 0 > s$), so $z = s + a = s + \sigma - s = \sigma$. The candidate post-order levels are just $\Sigma \cup \{\hat\sigma\} = \Sigma$.

For $0 \leq s < n$: $a = \max(0, \sigma - s)$, so $z = s + \max(0, \sigma - s) = \max(s, \sigma)$. Candidates are $\{\max(s, \sigma) : \sigma \in \Sigma\} = \{s, s+1, \ldots, n\}$ (since for $\sigma \leq s$, $\max(s, \sigma) = s$, and for $\sigma > s$, $\max(s, \sigma) = \sigma$). Plus $\max(s, \hat\sigma)$ which is already in this set.

For $s \geq n$: $a = 0$ for all $\sigma \in \Sigma$, so $z = s$ for all candidates.

This means we can simplify the inner loop:

```python
for i, s in enumerate(states):
    s_int = int(s)
    
    if s_int >= n:
        # All candidates give z = s
        z_idx = s_int + s_max
        alpha_new_restricted[i] = epsilon[i] + gamma * H_alpha[z_idx] + H_delta[z_idx]
    elif s_int >= 0:
        # Candidate z values: {s, s+1, ..., n}
        z_lo = s_int + s_max
        z_hi = n + s_max
        combo_vals = gamma * H_alpha[z_lo:z_hi+1] + H_delta[z_lo:z_hi+1]
        alpha_new_restricted[i] = epsilon[i] + np.max(combo_vals)
    else:
        # s < 0: candidate z values are Sigma = {0, 1, ..., n}
        combo_vals = gamma * H_alpha[np.array(Sigma) + s_max] + H_delta[np.array(Sigma) + s_max]
        alpha_new_restricted[i] = epsilon[i] + np.max(combo_vals)
```

### Convergence

The restricted action set variant uses the same operator structure with $A(s)$ being a subset of $\mathcal{A}$, so it's still a $\gamma$-contraction and converges at the same rate.

### What to Compute and Plot

1. Run the `alpha_restricted` fixed-point to convergence (same tolerance as before).

2. Verify: `alpha_max[s] <= alpha_restricted[s] <= alpha_sup[s]` for all $s$. This must hold since $\{\pi^\star(s), \hat\pi^\star(s)\} \subseteq A(s) \subseteq \mathcal{A}$.

3. Verify: `2 * alpha_restricted[s] >= gap[s]` for all $s$ (the bound is valid).

4. Report the same comparison table as before but with four rows:
   - True gap
   - `2 * alpha_max` (requires $\pi^\star$)
   - `2 * alpha_restricted` (requires only structural knowledge)
   - `2 * alpha_sup` (requires nothing)
   - Weighted-norm bound
   - Sup-norm bound

5. In the plots, add `alpha_restricted` as an additional curve (or replace `alpha_max` with `alpha_restricted` since the latter is the practically computable one).

### Key Point

The restricted action set exploits **structural knowledge about the policy class** (base-stock optimality) rather than knowledge of the optimal policy itself. The base-stock structure is a classical result from inventory theory that holds for this class of models regardless of the specific parameter values. The bound on $\sigma^\star \leq n$ comes from the demand distribution's support, not from solving the true model.
