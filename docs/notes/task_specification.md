# Task Specification: Numerical Experiments for Sample-Path Dependent AIS Bounds

## 1. Context and Motivation

### 1.1 The Problem Setting: POMDPs and Approximate Solutions

In a **partially observable Markov decision process (POMDP)**, a controller makes sequential decisions based on a growing history of observations and actions, without direct access to the system state. Finding optimal policies is computationally intractable (PSPACE-hard). A practical approach is to compress the history into a low-dimensional **approximate information state (AIS)** and solve an approximate dynamic program over this compressed representation.

### 1.2 The Original AIS Framework (Subramanian et al., JMLR 2022)

The AIS framework compresses the history $h_t = (y_{1:t}, a_{1:t-1})$ into $z_t = \sigma_t(h_t) \in \mathcal{Z}$ using an encoder $\sigma_t$. An **AIS generator** $(\sigma, \hat{c}, \hat{P})$ consists of:
- History compression functions $\sigma_t : \mathcal{H}_t \to \mathcal{Z}$
- Cost approximators $\hat{c}_t : \mathcal{Z} \times \mathcal{A} \to \mathbb{R}$
- Dynamics approximators $\hat{P}_t : \mathcal{Z} \times \mathcal{A} \to \Delta(\mathcal{Z})$

The AIS generator defines an approximate MDP on $\mathcal{Z}$ with value functions:
$$\hat{Q}^\star_t(z_t, a_t) = \hat{c}_t(z_t, a_t) + \int \hat{P}_t(dz_{t+1}|z_t, a_t) \hat{V}^\star_{t+1}(z_{t+1}), \quad \hat{V}^\star_t(z_t) = \inf_a \hat{Q}^\star_t(z_t, a)$$

The AIS-based policy acts on histories via: $\pi_t(h_t) = \hat{\pi}^\star_t(\sigma_t(h_t))$.

**Original AIS Theorem**: If the generator satisfies **uniform** bounds for all $h_t$ and $a$:
- (AP1): $|c_t(h_t, a) - \hat{c}_t(\sigma_t(h_t), a)| \leq \bar{\varepsilon}_t$ (cost approximation)
- (AP2): $d_{\mathfrak{F}}(\nu_t(\cdot|h_t, a), \hat{P}_t(\cdot|\sigma_t(h_t), a)) \leq \bar{\delta}_t$ (transition approximation)

where $\nu_t(\cdot|h_t, a) = P(\sigma_{t+1}(H_{t+1}) \in \cdot | h_t, a)$ is the true conditional distribution of the next AIS value, then:
$$V^\pi_t(h_t) - V^\star_t(h_t) \leq 2\bar{\alpha}_t$$
where $\bar{\alpha}_t = \bar{\varepsilon}_t + \sum_{\tau=t}^{T-1}[\bar{\delta}_\tau \rho_{\mathfrak{F}}(\hat{V}_{\tau+1}) + \bar{\varepsilon}_{\tau+1}]$.

Here $d_{\mathfrak{F}}$ is an integral probability metric (IPM) and $\rho_{\mathfrak{F}}(g) = \inf\{\lambda > 0 : g/\lambda \in \mathfrak{F}\}$ is the Minkowski functional.

**Limitation**: The bound $2\bar{\alpha}_t$ is a single constant driven by the worst-case history-action pair. If the AIS is accurate for typical histories but poor on rare ones, the bound is loose.

### 1.3 Our Current Paper: Sample-Path Dependent AIS Bounds

We generalize the AIS framework by allowing the approximation errors to depend on the realized history and action. The three assumptions are:

**Assumption 2** (cost approximation): For each $t$, there exists $\varepsilon_t : \mathcal{H}_t \times \mathcal{A} \to \mathbb{R}_{\geq 0}$ such that for all $h_t$ and $a$:
$$\left|\int c_t(s_t, a) P(ds_t|h_t) - \hat{c}_t(\sigma_t(h_t), a)\right| \leq \varepsilon_t(h_t, a)$$

**Assumption 3** (transition approximation, direct form): For each $t$, there exists $\Delta_t : \mathcal{H}_t \times \mathcal{A} \to \mathbb{R}_{\geq 0}$ such that for all $h_t$ and $a$:
$$\left|\int \hat{V}_{t+1}(z_{t+1}) P(dz_{t+1}|h_t, a) - \int \hat{V}_{t+1}(z_{t+1}) \hat{P}(dz_{t+1}|\sigma_t(h_t), a)\right| \leq \Delta_t(h_t, a)$$

**Assumption 4** (transition approximation, IPM form — sufficient for Assumption 3): For each $t$, there exists $\delta_t : \mathcal{H}_t \times \mathcal{A} \to \mathbb{R}_{\geq 0}$ such that for all $h_t$ and $a$:
$$d_{\mathfrak{F}}(\nu_t(\cdot|h_t, a), \hat{P}_t(\cdot|\sigma_t(h_t), a)) \leq \delta_t(h_t, a)$$

Under Assumption 4, Assumption 3 holds with $\Delta_t(h_t, a) = \rho_{\mathfrak{F}}(\hat{V}_{t+1}) \cdot \delta_t(h_t, a)$.

**Theorem 2** (Main result): Under Assumptions 2 and 3, the AIS-based policy satisfies for all $t$ and $h_t$:
$$|V^\pi_t(h_t) - V^\star_t(h_t)| \leq 2\alpha_t(h_t)$$
where $\beta_T(h_T, a_T) = \varepsilon_T(h_T, a_T)$ and for $t < T$:
$$\beta_t(h_t, a_t) = \varepsilon_t(h_t, a_t) + \int \alpha_{t+1}(h_t, a_t, y_{t+1}) P(dy_{t+1}|h_t, a_t) + \Delta_t(h_t, a_t)$$
and
$$\alpha_t(h_t) = \max\{\beta_t(h_t, \pi^\star_t(h_t)),\; \beta_t(h_t, \hat{\pi}^\star_t(\sigma_t(h_t)))\} \leq \sup_{a \in \mathcal{A}} \beta_t(h_t, a)$$

**Corollary 1**: Under Assumptions 2 and 4, the result holds with $\Delta_t(h_t, a_t)$ replaced by $\rho_{\mathfrak{F}}(\hat{V}_{t+1}) \delta_t(h_t, a_t)$.

**Remark**: If (AP1) and (AP2) of the original AIS theorem hold (uniform bounds), then Assumptions 2 and 4 hold with $\varepsilon_t(h_t, a) = \bar{\varepsilon}_t$ and $\delta_t(h_t, a) = \bar{\delta}_t$. The original AIS bound is an immediate consequence.

**Key structural differences from the original AIS bound**:
1. The future error $\alpha_{t+1}$ is propagated through the **true** transition kernel, weighting by reachability.
2. The $\alpha$-$\beta$ separation allows $\alpha_t(h_t)$ to only dominate $\beta_t$ at the two relevant actions ($\pi^\star_t(h_t)$ and $\hat{\pi}^\star_t(\sigma_t(h_t))$), not all actions.
3. The bound is **pointwise** in $h_t$, not a single constant.

### 1.4 MDP Specialization (Corollary for Experiments)

When the state is fully observed ($Y_t = S_t$), histories reduce to states, $\sigma_t(h_t) = s_t$, $\mathcal{Z} = \mathcal{S}$, and the AIS model is simply the approximate MDP $\hat{\mathcal{M}}$. Everything becomes state-dependent rather than history-dependent.

**MDP Corollary**: Under:
- $|c(s, a) - \hat{c}(s, a)| \leq \varepsilon(s, a)$ for all $(s, a)$
- $|\sum_{s'} \hat{V}^\star(s') P(s'|s,a) - \sum_{s'} \hat{V}^\star(s') \hat{P}(s'|s,a)| \leq \Delta(s, a)$ for all $(s, a)$

The policy $\hat{\pi}^\star$ satisfies $|V^{\hat{\pi}^\star}(s) - V^\star(s)| \leq 2\alpha(s)$ where:
$$\beta(s, a) = \varepsilon(s, a) + \gamma \sum_{s'} \alpha(s') P(s'|s, a) + \Delta(s, a)$$
$$\alpha(s) = \max\{\beta(s, \pi^\star(s)),\; \beta(s, \hat{\pi}^\star(s))\} \leq \sup_a \beta(s, a)$$

For infinite horizon, this becomes a fixed-point equation solved by iteration.

### 1.5 The Model Approximation Paper (Bozkurt et al., IEEE TAC 2025)

This earlier paper by the same authors considers the same question — bounding $\|V^{\hat{\pi}^\star} - V^\star\|$ — for infinite-horizon discounted MDPs, using **Bellman mismatch functionals** and **weighted norms**.

**Setup**: Two MDPs $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, c, \gamma \rangle$ and $\hat{\mathcal{M}} = \langle \mathcal{S}, \mathcal{A}, \hat{P}, \hat{c}, \gamma \rangle$ on the same state/action spaces with discount factor $\gamma \in (0,1)$.

**Weighted norm**: For weight function $w : \mathcal{S} \to [1, \infty)$:
$$\|v\|_w = \sup_{s \in \mathcal{S}} \frac{|v(s)|}{w(s)}$$

**Bellman operators**: For policy $\pi$:
$$[\mathcal{B}^\pi v](s) = c_\pi(s) + \gamma \int v(s') P_\pi(ds'|s)$$

**Bellman mismatch functionals**:
- Policy mismatch: $\mathcal{D}^{\pi,\hat{\pi}} v = \|\mathcal{B}^\pi v - \hat{\mathcal{B}}^{\hat{\pi}} v\|_w$
- Same-policy mismatch: $\mathcal{D}^\pi v = \|\mathcal{B}^\pi v - \hat{\mathcal{B}}^\pi v\|_w$
- Optimality mismatch: $\mathcal{D}^\star v = \|\mathcal{B}^\star v - \hat{\mathcal{B}}^\star v\|_w$
- Maximum mismatch: $\mathcal{D}^{\max} v = \sup_{a \in \mathcal{A}} \|\mathcal{B}^{\pi_a} v - \hat{\mathcal{B}}^{\pi_a} v\|_w$ (where $\pi_a$ is the open-loop policy selecting $a$ always)

**Stability condition**: A policy $\pi$ is $(\kappa, w)$-stable if $\int w(s') P_\pi(ds'|s) \leq \kappa \cdot w(s)$ for all $s$, with $\gamma\kappa < 1$.

**Key results from that paper**:

*Theorem 2* (Main bound under Assumptions 1-3): Under DP solvability, stability of relevant policies, and greedy policy stability:
$$\|V^{\hat{\pi}^\star} - V^\star\|_w \leq \frac{1}{1-\gamma\kappa}[\mathcal{D}^{\hat{\pi}^\star}(\hat{V}^\star) + \mathcal{D}^\star(\hat{V}^\star)]$$

This translates to a pointwise bound: $V^{\hat{\pi}^\star}(s) - V^\star(s) \leq \frac{1}{1-\gamma\kappa}[\mathcal{D}^{\hat{\pi}^\star}(\hat{V}^\star) + \mathcal{D}^\star(\hat{V}^\star)] \cdot w(s)$.

*Theorem 3* (Open-loop stability bound):
$$\|V^{\hat{\pi}^\star} - V^\star\|_w \leq \frac{2}{1-\gamma\kappa} \mathcal{D}^{\max}(\hat{V}^\star)$$

*Sup-norm special case* ($w \equiv 1$, $\kappa = 1$):
$$\|V^{\hat{\pi}^\star} - V^\star\|_\infty \leq \frac{2}{1-\gamma} \mathcal{D}^{\max}(\hat{V}^\star)$$

**IPM-based bounds** (Theorem 5, part 4 of that paper): The maximum mismatch can be bounded as:
$$\mathcal{D}^{\max}_{\boldsymbol{\alpha}} v \leq \varepsilon^{\max}_{\boldsymbol{\alpha}} + \gamma \rho_{\mathfrak{F}}(v) \delta^{\max}_{\mathfrak{F}}$$
where $\varepsilon^{\max}_{\boldsymbol{\alpha}} = \sup_{s,a} |\alpha_1 c(s,a) + \alpha_2 - \hat{c}(s,a)|/w(s)$ and $\delta^{\max}_{\mathfrak{F}} = \sup_{s,a} d_{\mathfrak{F}}(P(\cdot|s,a), \hat{P}(\cdot|s,a))/w(s)$.

### 1.6 Relationship Between the Three Papers

| Aspect | Original AIS | Sample-Path AIS (ours) | Model Approximation |
|--------|-------------|----------------------|-------------------|
| Setting | POMDP, finite horizon | POMDP, finite horizon | MDP, infinite horizon discounted |
| State space | History $\mathcal{H}_t$ vs AIS $\mathcal{Z}$ | History $\mathcal{H}_t$ vs AIS $\mathcal{Z}$ | Same state space $\mathcal{S}$ |
| Errors | Uniform $\bar{\varepsilon}_t, \bar{\delta}_t$ | History-dependent $\varepsilon_t(h_t, a_t), \delta_t(h_t, a_t)$ | Weighted-norm sup over $(s,a)$ |
| Bound type | Single constant $2\bar{\alpha}_t$ | Pointwise $2\alpha_t(h_t)$ | Weighted norm $\|V^{\hat{\pi}^\star} - V^\star\|_w$ |
| Error propagation | Sum (no dynamics weighting) | Through true dynamics $P$ | Geometric series via $1/(1-\gamma\kappa)$ |

The experiment compares the MDP specialization of our result with the model approximation paper's bounds on the **same** inventory management example.

---

## 2. Experimental Setup

### 2.1 Inventory Management MDP

**State space**: $\mathcal{S} = \{-S_{\max}, -S_{\max}+1, \ldots, S_{\max}\}$ with $S_{\max} = 500$ (so $|\mathcal{S}| = 1001$).

**Action space**: $\mathcal{A} = \{0, 1, \ldots, S_{\max}\}$ (so $|\mathcal{A}| = 501$).

**Dynamics**: $S_{t+1} = [S_t + A_t - N_t]^{S_{\max}}_{-S_{\max}}$ where $[\cdot]$ clips to $[-S_{\max}, S_{\max}]$ and demand $N_t \sim \text{Binomial}(n, q)$ i.i.d.

**Per-step cost**: $c(s, a) = pa + c_h s \mathbf{1}_{\{s \geq 0\}} - c_s s \mathbf{1}_{\{s < 0\}}$.

**True model**: $\mathcal{M} = (S_{\max}=500, \gamma=0.75, n=10, q=0.4, c_h=4.0, c_s=2, p=5)$

**Approximate model**: $\hat{\mathcal{M}} = (S_{\max}=500, \gamma=0.75, n=10, q=0.5, c_h=3.8, c_s=2, p=5)$

The models differ in:
- Demand probability: $q = 0.4$ (true) vs $q = 0.5$ (approx) — the approximate model overestimates demand
- Holding cost: $c_h = 4.0$ (true) vs $c_h = 3.8$ (approx) — the approximate model underestimates holding cost

**Optimal policy structure**: Both models have base-stock optimal policies $\pi^\star(s) = \max(0, \sigma - s)$. For $\hat{\mathcal{M}}$, $\sigma = 2$. Since demand support is $\{0, 1, \ldots, 10\}$, after transient, inventory stays in $\{-8, \ldots, 2\}$.

### 2.2 Weight Function and Stability (for existing bounds)

**Weight function**: $w(s) = 1 + \ell [\hat{c}_h s \mathbf{1}_{\{s \geq 0\}} - \hat{c}_s s \mathbf{1}_{\{s < 0\}}]$ with $\ell = 1.5 \times 10^{-2}$.

**Stability constant**: $\kappa = 1.07$, verified numerically.

**Family of weight functions** (for the multi-weight experiment): $\mathcal{W} = \{1 + \ell \bar{c}(s) : \ell \in \{0, 0.5 \times 10^{-2}, 10^{-2}, \ldots, 2.5 \times 10^{-2}\}\}$ where $\bar{c}(s) = \hat{c}_h s \mathbf{1}_{\{s \geq 0\}} - \hat{c}_s s \mathbf{1}_{\{s < 0\}}$.

---

## 3. The Three Bounds to Compare

### Bound 1: Sup-norm bound (existing, from model approximation paper)

Set $w(s) \equiv 1$ (so $\kappa_w$ satisfies $\gamma \kappa_w < 1$). The bound is a single constant:

$$V^{\hat{\pi}^\star}(s) - V^\star(s) \leq \frac{2}{1-\gamma\kappa_w} \mathcal{D}^{\max}(\hat{V}^\star) \quad \text{for all } s$$

where:
$$\mathcal{D}^{\max}(\hat{V}^\star) = \max_{s \in \mathcal{S}, a \in \mathcal{A}} \left|c(s,a) + \gamma \sum_{s'} \hat{V}^\star(s') P(s'|s,a) - \hat{c}(s,a) - \gamma \sum_{s'} \hat{V}^\star(s') \hat{P}(s'|s,a)\right|$$

This appears as a horizontal line on the plots.

### Bound 2: Weighted-norm bound (existing, from model approximation paper)

Using weight function $w(s)$ and stability constant $\kappa$:

$$V^{\hat{\pi}^\star}(s) - V^\star(s) \leq \frac{1}{1-\gamma\kappa}[\mathcal{D}^{\hat{\pi}^\star}(\hat{V}^\star) + \mathcal{D}^\star(\hat{V}^\star)] \cdot w(s)$$

where:
$$\mathcal{D}^{\hat{\pi}^\star}(\hat{V}^\star) = \max_{s \in \mathcal{S}} \frac{|c(s, \hat{\pi}^\star(s)) + \gamma \sum_{s'} \hat{V}^\star(s') P(s'|s, \hat{\pi}^\star(s)) - \hat{c}(s, \hat{\pi}^\star(s)) - \gamma \sum_{s'} \hat{V}^\star(s') \hat{P}(s'|s, \hat{\pi}^\star(s))|}{w(s)}$$

$$\mathcal{D}^\star(\hat{V}^\star) = \max_{s \in \mathcal{S}} \frac{|\min_a [c(s,a) + \gamma \sum_{s'} \hat{V}^\star(s') P(s'|s,a)] - \min_a [\hat{c}(s,a) + \gamma \sum_{s'} \hat{V}^\star(s') \hat{P}(s'|s,a)]|}{w(s)}$$

This bound is proportional to $w(s)$.

### Bound 3: Sample-path dependent bound (NEW — to implement)

$$V^{\hat{\pi}^\star}(s) - V^\star(s) \leq 2\alpha(s)$$

Computed via fixed-point iteration. Initialize $\alpha^{(0)}(s) = 0$ for all $s$. Iterate:

**Step A**: For each $(s, a)$, compute:
$$\varepsilon(s, a) = |c(s, a) - \hat{c}(s, a)|$$
$$\Delta(s, a) = \left|\sum_{s'} \hat{V}^\star(s') P(s'|s,a) - \sum_{s'} \hat{V}^\star(s') \hat{P}(s'|s,a)\right|$$

These are computed once before the iteration.

**Step B**: Iterate until convergence:
$$\beta^{(k)}(s, a) = \varepsilon(s, a) + \gamma \sum_{s'} \alpha^{(k)}(s') P(s'|s,a) + \Delta(s, a)$$
$$\alpha^{(k+1)}(s) = \max\{\beta^{(k)}(s, \pi^\star(s)),\; \beta^{(k)}(s, \hat{\pi}^\star(s))\}$$

Convergence criterion: $\max_s |\alpha^{(k+1)}(s) - \alpha^{(k)}(s)| < \texttt{tol}$ (e.g., $10^{-10}$).

**Also compute the coarser sup-over-actions version**:
$$\alpha^{(k+1)}_{\sup}(s) = \max_a \beta^{(k)}(s, a)$$

This allows comparing $\alpha_{\max}$ (using only two relevant actions) vs $\alpha_{\sup}$ (using all actions).

---

## 4. Task Phases

### Phase 1: Understand and Audit the Existing Codebase

The existing codebase generates figures for the model approximation paper (Bozkurt et al., IEEE TAC 2025). You need to:

1. **Read the code thoroughly** and create a mapping:
   - Which functions/scripts compute $V^\star$, $\hat{V}^\star$, $\pi^\star$, $\hat{\pi}^\star$ (value iteration)
   - Which functions compute $V^{\hat{\pi}^\star}$ (policy evaluation)
   - Which functions compute the Bellman mismatch functionals $\mathcal{D}^{\hat{\pi}^\star}$, $\mathcal{D}^\star$, $\mathcal{D}^{\max}$
   - Which functions verify stability ($\kappa$ computation)
   - Which functions construct weight functions
   - Which scripts generate Figures 1-6 of the paper

2. **Run the existing code** and verify the output matches the paper's figures. Save these outputs as the baseline.

3. **Document the mapping** in a brief README: function → paper equation/theorem → figure.

### Phase 2: Refactor and Clean Up

Refactor into a clean modular structure:

```
project/
├── models.py                      # Model definitions
├── solvers.py                     # Value iteration, policy evaluation
├── bounds.py                      # All bound computations
├── plots.py                       # Plotting utilities
├── run_existing_experiments.py    # Reproduces model approx paper figures
├── run_new_experiments.py         # New comparison experiments
├── figures/
│   ├── existing/                  # Reproduced figures from model approx paper
│   └── comparison/                # New comparison figures
├── tests.py                       # Sanity checks
└── README.md                      # Documentation
```

**`models.py`**: Define an `InventoryMDP` class with:
- Parameters: $S_{\max}$, $\gamma$, $n$, $q$, $c_h$, $c_s$, $p$
- Methods to compute: transition matrix $P(s'|s,a)$ (sparse or dense), cost matrix $c(s,a)$
- Pre-compute and cache transition matrices (these are reused many times)

**`solvers.py`**: 
- `value_iteration(model, tol)` → returns $V^\star$, $\pi^\star$
- `policy_evaluation(model, policy, tol)` → returns $V^\pi$
- Both should work with the transition matrix representation

**`bounds.py`**:
- `sup_norm_bound(M, M_hat, V_hat_star)` → scalar
- `weighted_norm_bound(M, M_hat, V_hat_star, pi_star, pi_hat_star, w, kappa)` → array over states
- `sample_path_bound(M, M_hat, V_hat_star, pi_star, pi_hat_star, tol)` → returns `alpha_max`, `alpha_sup` arrays
- `compute_kappa(model, w)` → stability constant
- `bellman_mismatch_functionals(M, M_hat, V_hat_star, pi_star, pi_hat_star, w)` → $\mathcal{D}^{\hat{\pi}^\star}$, $\mathcal{D}^\star$, $\mathcal{D}^{\max}$

**`tests.py`**: Sanity checks:
- $V^\star(s) \leq V^{\hat{\pi}^\star}(s)$ for all $s$
- Value iteration converges
- Optimal policies are base-stock
- All bounds are non-negative
- $2\alpha(s) \geq V^{\hat{\pi}^\star}(s) - V^\star(s)$ for all $s$ (the bound is valid)
- With constant errors, sample-path bound recovers the original AIS/sup-norm bound

**Critical requirement**: After refactoring, `run_existing_experiments.py` must reproduce the same figures as the original code. Compare numerically, not just visually.

### Phase 3: Implement New Bounds and Generate Comparison Plots

#### Step 3.1: Compute state-action-dependent errors

```python
# Cost mismatch: |c(s,a) - c_hat(s,a)| for each (s,a)
epsilon = np.abs(c - c_hat)  # shape (|S|, |A|)

# Transition mismatch on hat_V_star: for each (s,a)
# Delta[s,a] = |P[s,a,:] @ hat_V_star - P_hat[s,a,:] @ hat_V_star|
# where P[s,a,:] is the row of the transition matrix for (s,a)
Delta = np.abs(
    np.einsum('saj,j->sa', P, hat_V_star) - np.einsum('saj,j->sa', P_hat, hat_V_star)
)
# shape (|S|, |A|)
```

#### Step 3.2: Fixed-point iteration for alpha

```python
gamma = 0.75
alpha = np.zeros(num_states)
tol = 1e-10

for k in range(max_iter):
    # Propagate alpha through true dynamics: for each (s,a), compute gamma * sum_s' alpha(s') P(s'|s,a)
    propagated = gamma * np.einsum('saj,j->sa', P, alpha)  # shape (|S|, |A|)
    
    # beta(s,a) = epsilon(s,a) + propagated(s,a) + Delta(s,a)
    beta = epsilon + propagated + Delta  # shape (|S|, |A|)
    
    # alpha_max: evaluate beta only at pi_star(s) and pi_hat_star(s)
    beta_at_pi_star = beta[np.arange(num_states), pi_star]
    beta_at_pi_hat_star = beta[np.arange(num_states), pi_hat_star]
    alpha_new = np.maximum(beta_at_pi_star, beta_at_pi_hat_star)
    
    if np.max(np.abs(alpha_new - alpha)) < tol:
        print(f"Converged at iteration {k}")
        break
    alpha = alpha_new

alpha_max = alpha  # The tight version

# Also compute sup version
alpha_sup = np.max(beta, axis=1)  # sup over all actions
```

**Note on memory**: With $|\mathcal{S}| = 1001$ and $|\mathcal{A}| = 501$, the full transition matrix $P$ has shape $(1001, 501, 1001)$ which is about 4 GB in float64. You may need to:
- Use sparse matrices (demand is Binomial(10, q) so each row of $P$ has at most 11 nonzero entries)
- Or compute the matrix-vector products $P[s,a,:] @ \text{alpha}$ column by column
- Or reduce $S_{\max}$ for initial experiments

#### Step 3.3: Generate plots

**Plot 1: Shaded region comparison (matching paper style)**

Three panels side by side:
- Panel (a): Sup-norm — shade between $V^{\hat{\pi}^\star}(s)$ and $V^{\hat{\pi}^\star}(s) - \text{sup-norm bound}$
- Panel (b): Weighted-norm — shade between $V^{\hat{\pi}^\star}(s)$ and $V^{\hat{\pi}^\star}(s) - \text{weighted-norm bound}(s)$
- Panel (c): Sample-path — shade between $V^{\hat{\pi}^\star}(s)$ and $V^{\hat{\pi}^\star}(s) - 2\alpha(s)$

In each panel, also plot $V^\star(s)$ as a reference (it should lie inside the shaded region).

Do both full state space and zoomed-in ($s \in \{-10, \ldots, 10\}$) versions.

**Plot 2: Direct bound comparison**

On a single axes, plot as a function of $s$:
- $V^{\hat{\pi}^\star}(s) - V^\star(s)$ (true gap, black solid)
- $2\alpha_{\max}(s)$ (sample-path bound, max version, blue solid)
- $2\alpha_{\sup}(s)$ (sample-path bound, sup version, blue dashed)
- Weighted-norm bound at $s$ (red solid)
- Sup-norm bound (green horizontal line)

Zoomed into $s \in \{-10, \ldots, 10\}$.

**Plot 3: Error decomposition**

For $a = \hat{\pi}^\star(s)$, plot separately:
- $\varepsilon(s, \hat{\pi}^\star(s))$ (cost mismatch)
- $\Delta(s, \hat{\pi}^\star(s))$ (transition mismatch)
- $\gamma \sum_{s'} \alpha(s') P(s'|s, \hat{\pi}^\star(s))$ (propagated future error)
This shows which error source dominates at each state.

**Plot 4: Convergence of alpha iteration**

Plot $\max_s |\alpha^{(k+1)}(s) - \alpha^{(k)}(s)|$ vs iteration $k$.

**Plot 5: Max vs sup comparison**

Plot $2\alpha_{\max}(s)$ and $2\alpha_{\sup}(s)$ on the same axes in the operating region, showing the benefit of the tighter characterization.

---

## 5. Expected Outcomes

1. **In the operating region** $\{-8, \ldots, 2\}$: The sample-path bound $2\alpha(s)$ should be significantly tighter than both the sup-norm and weighted-norm bounds. This is because $\alpha(s)$ propagates errors through the true dynamics, which concentrate in this region.

2. **At extreme states**: All three bounds may be comparable, since errors $\varepsilon(s,a)$ and $\Delta(s,a)$ are large there.

3. **Max vs sup**: $\alpha_{\max}(s) \leq \alpha_{\sup}(s)$ always, and the gap should be visible, demonstrating the practical benefit of evaluating $\beta$ only at the two relevant actions.

4. **Recovery of original bound**: With constant errors, the sample-path iteration should converge to $\alpha(s) = \bar{\alpha}$ (constant), recovering the sup-norm bound.

5. **Convergence**: The $\alpha$ iteration should converge geometrically since the operator is a $\gamma$-contraction.

---

## 6. Important Implementation Notes

1. **Transition matrices are sparse**: Demand $N_t \sim \text{Binomial}(10, q)$ has support $\{0, \ldots, 10\}$, so $P(s'|s,a)$ has at most 11 nonzero entries per $(s,a)$. Use `scipy.sparse` if memory is an issue.

2. **Clipping**: The dynamics clip to $[-S_{\max}, S_{\max}]$. At the boundaries, probability mass accumulates. Make sure $P(s'|s,a)$ sums to 1 for each $(s,a)$.

3. **Policy representation**: For a base-stock policy with level $\sigma$: $\pi(s) = \max(0, \sigma - s)$, but also clipped to $[0, S_{\max}]$.

4. **Discount factor**: $\gamma = 0.75$, so $\gamma^{100} \approx 10^{-12.5}$. The fixed-point iteration should converge quickly.

5. **Numerical precision**: Use `tol = 1e-10` for value iteration and the $\alpha$ fixed-point. Check that $|V^\star(s) - V^{\hat{\pi}^\star}(s)|$ is always $\leq 2\alpha(s) + \text{tol}$ (the bound should be valid up to numerical noise).

6. **Existing paper figures**: The model approximation paper has the following key figures for the inventory example:
   - Figure 1: Weighted-norm vs sup-norm shaded regions (full state space)
   - Figure 2: Zoomed-in versions of Figure 1
   - Figure 3: Multiple weight functions comparison
   - Figure 4: Bounds from $(\alpha_1, \alpha_2)$ cost transformation
   - Figure 5: Model stability vs policy stability comparison
   
   All of these should be reproducible from the existing codebase before implementing new bounds.
