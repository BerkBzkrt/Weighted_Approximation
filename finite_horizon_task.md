# Finite-Horizon Inventory Control: Sample-Path Bound Implementation

## Objective

Implement a **finite-horizon** version of the sample-path dependent AIS bound for the inventory control example. This version applies Corollary 1 (MDP specialization) directly, without any infinite-horizon extension or contraction arguments. The discount factor $\gamma$ is absorbed into the per-step cost, yielding a standard finite-horizon undiscounted MDP.

**Important:** Create a separate notebook (`notebooks/Finite_Horizon_Bounds.ipynb`) for all finite-horizon computations. Do not modify the existing infinite-horizon code or notebooks — we may keep both analyses in the paper.

---

## Background: Why Finite Horizon

The theoretical results (Corollary 1 in the paper) are stated for finite-horizon MDPs. The existing numerical experiments use an infinite-horizon discounted formulation, which required additional arguments (fixed-point iteration, contraction remark) beyond the scope of the stated theorem. The finite-horizon version eliminates this gap: the $\alpha$-$\beta$ recursion is run backward from $t = T$ to $t = 1$ exactly as stated in Corollary 1.

---

## The Finite-Horizon Formulation

### Absorbing the discount factor into the cost

The infinite-horizon discounted objective is:

$$\min_\pi \mathbb{E}\left[\sum_{t=1}^{\infty} \gamma^{t-1} c(s_t, a_t)\right]$$

We reformulate this as a finite-horizon undiscounted MDP with time-varying costs:

$$\min_\pi \mathbb{E}\left[\sum_{t=1}^{T} c_t(s_t, a_t)\right]$$

where $c_t(s, a) = \gamma^{t-1} c(s, a) = \gamma^{t-1}[pa + h(s)]$.

Similarly for the approximate model: $\hat{c}_t(s, a) = \gamma^{t-1} \hat{c}(s, a) = \gamma^{t-1}[pa + \hat{h}(s)]$.

The transitions are stationary (time-independent): $P_t = P$ and $\hat{P}_t = \hat{P}$ for all $t$.

### Horizon choice

With $\gamma = 0.75$, the truncation error from cutting off at horizon $T$ is at most $\gamma^T \cdot \|c\|_\infty / (1-\gamma)$. For $T = 100$, $\gamma^{100} \approx 3 \times 10^{-13}$, so the truncation error is negligible. Use $T = 100$.

### Key point: no $\gamma$ in the $\beta$ recursion

In the finite-horizon formulation, the $\beta$ recursion from Corollary 1 is:

$$\beta_t(s, a) = \varepsilon_t(s, a) + \sum_{s'} \alpha_{t+1}(s') P(s'|s, a) + \Delta_t(s, a)$$

There is **no** $\gamma$ multiplying the $\sum_{s'} \alpha_{t+1}$ term. The discounting is entirely absorbed into $\varepsilon_t$ and $\Delta_t$ through the time-varying costs and value functions.

---

## Step-by-Step Computation

### Step 0: Solve both models via backward induction

For both the true model $M$ and approximate model $\hat{M}$, compute the finite-horizon value functions and optimal policies by backward induction.

**Terminal condition:** $V^\star_{T+1}(s) = 0$ and $\hat{V}^\star_{T+1}(s) = 0$ for all $s$.

**Backward step** (for $t = T, T-1, \ldots, 1$):

For the true model:
$$Q^\star_t(s, a) = c_t(s, a) + \sum_{s'} V^\star_{t+1}(s') P(s'|s, a)$$
$$V^\star_t(s) = \min_a Q^\star_t(s, a), \quad \pi^\star_t(s) = \arg\min_a Q^\star_t(s, a)$$

For the approximate model:
$$\hat{Q}^\star_t(s, a) = \hat{c}_t(s, a) + \sum_{s'} \hat{V}^\star_{t+1}(s') \hat{P}(s'|s, a)$$
$$\hat{V}^\star_t(s) = \min_a \hat{Q}^\star_t(s, a), \quad \hat{\pi}^\star_t(s) = \arg\min_a \hat{Q}^\star_t(s, a)$$

Where $c_t(s,a) = \gamma^{t-1}[pa + h(s)]$ and $\hat{c}_t(s,a) = \gamma^{t-1}[pa + \hat{h}(s)]$.

**Note:** The H-array trick applies at each time step since the transitions are stationary. The H-arrays for the transition-related quantities change at each $t$ (because $\hat{V}^\star_{t+1}$ and $\alpha_{t+1}$ change), but the demand PMFs $W_M$ and $W_{\hat{M}}$ are the same at every step.

**Store:** $\hat{V}^\star_t(s)$ and $\hat{\pi}^\star_t(s)$ for all $t$ and $s$ (needed for the $\alpha$-$\beta$ recursion).

### Step 1: Policy evaluation of $\hat{\pi}^\star$ in the true model

Compute $V^{\hat{\pi}^\star}_t(s)$ for all $t$ and $s$ by evaluating the approximate model's policy $\hat{\pi}^\star_t$ under the true model's dynamics and costs.

**Terminal:** $V^{\hat{\pi}^\star}_{T+1}(s) = 0$.

**Backward step:**
$$V^{\hat{\pi}^\star}_t(s) = c_t(s, \hat{\pi}^\star_t(s)) + \sum_{s'} V^{\hat{\pi}^\star}_{t+1}(s') P(s'|s, \hat{\pi}^\star_t(s))$$

The true sub-optimality gap at time 1 is: $\text{gap}(s) = V^{\hat{\pi}^\star}_1(s) - V^\star_1(s)$.

### Step 2: Compute time-varying error quantities

For each $t = 1, \ldots, T$ and each $(s, a)$:

**Cost mismatch:**
$$\varepsilon_t(s, a) = |c_t(s, a) - \hat{c}_t(s, a)| = \gamma^{t-1} |h_M(s) - h_{\hat{M}}(s)|$$

Since $p$ is identical in both models, $\varepsilon_t(s, a) = \gamma^{t-1} \varepsilon(s)$ where $\varepsilon(s) = |h_M(s) - h_{\hat{M}}(s)|$ is the same state-only mismatch as before, just scaled by $\gamma^{t-1}$.

**Transition mismatch:**
$$\Delta_t(s, a) = \left|\sum_{s'} \hat{V}^\star_{t+1}(s') P(s'|s,a) - \sum_{s'} \hat{V}^\star_{t+1}(s') \hat{P}(s'|s,a)\right|$$

Note: $\Delta_t$ depends on $\hat{V}^\star_{t+1}$, which varies with $t$. So $\Delta_t$ must be recomputed at each time step (unlike the infinite-horizon case where it was computed once).

**With the H-array trick:** At each time $t$, build:
$$H^{(t)}_M[z] = \sum_{w=0}^n W_M[w] \cdot \hat{V}^\star_{t+1}(\text{clip}(z - w))$$
$$H^{(t)}_{\hat{M}}[z] = \sum_{w=0}^n W_{\hat{M}}[w] \cdot \hat{V}^\star_{t+1}(\text{clip}(z - w))$$
$$H^{(t)}_\Delta[z] = |H^{(t)}_M[z] - H^{(t)}_{\hat{M}}[z]|$$

Then $\Delta_t(s, a) = H^{(t)}_\Delta[s + a]$.

### Step 3: Backward $\alpha$-$\beta$ recursion

**Terminal:** $\beta_T(s, a) = \varepsilon_T(s, a) = \gamma^{T-1} \varepsilon(s)$, and $\alpha_T(s) = \max_{a \in A(s)} \beta_T(s, a) = \gamma^{T-1} \varepsilon(s)$ (since $\varepsilon$ doesn't depend on $a$).

**Backward step** (for $t = T-1, T-2, \ldots, 1$):

1. Build the propagation H-array for $\alpha_{t+1}$:
$$H^{(t)}_\alpha[z] = \sum_{w=0}^n W_M[w] \cdot \alpha_{t+1}(\text{clip}(z - w))$$

2. Compute $\beta_t(s, a)$:
$$\beta_t(s, a) = \varepsilon_t(s) + H^{(t)}_\alpha[s + a] + H^{(t)}_\Delta[s + a]$$

Note: **no $\gamma$** multiplying $H^{(t)}_\alpha$. The $\gamma$ is already in $\varepsilon_t$.

3. Update $\alpha_t(s)$. Three variants:

**alpha_max** (requires $\pi^\star_t$):
$$\alpha_t(s) = \max\{\beta_t(s, \pi^\star_t(s)), \; \beta_t(s, \hat{\pi}^\star_t(s))\}$$

**alpha_restricted** (requires only structural knowledge, $\Sigma = \{0, \ldots, 10\}$):
$$\alpha_t(s) = \max_{a \in A_{\text{res}}(s)} \beta_t(s, a)$$
where $A_{\text{res}}(s) = \{\max(0, \sigma - s) : \sigma \in \Sigma\} \cup \{\hat{\pi}^\star_t(s)\}$.

Note: For the restricted variant, the action set now includes $\hat{\pi}^\star_t(s)$ which is time-varying. However, for large enough $t$ (past the transient), $\hat{\pi}^\star_t(s)$ will be the same base-stock policy as the stationary one, so in practice this barely matters.

**alpha_sup** (no structural knowledge needed):
$$\alpha_t(s) = \max_{a \in \mathcal{A}} \beta_t(s, a)$$

### Step 4: The bound

The sample-path bound at time $t = 1$ is:
$$V^{\hat{\pi}^\star}_1(s) - V^\star_1(s) \leq 2\alpha_1(s)$$

This is the quantity to compare against the infinite-horizon sup-norm and weighted-norm bounds.

---

## Comparison with Existing Bounds

The sup-norm and weighted-norm bounds from the model approximation paper are derived for the infinite-horizon discounted setting and do not have direct finite-horizon counterparts. However, since $\gamma^T < 10^{-12}$ for $T = 100$, the finite-horizon value functions at $t = 1$ agree with their infinite-horizon counterparts to numerical precision. The comparison is therefore:

- **Sample-path bound:** $2\alpha_1(s)$ from the finite-horizon backward recursion (our result, directly from Corollary 1)
- **Sup-norm bound:** computed from the infinite-horizon formulation as before (constant over all $s$)
- **Weighted-norm bound:** computed from the infinite-horizon formulation as before ($C \cdot w(s)$)
- **True gap:** $V^{\hat{\pi}^\star}_1(s) - V^\star_1(s) \approx V^{\hat{\pi}^\star}(s) - V^\star(s)$ (the infinite-horizon gap)

All three bounds are upper bounds on the same quantity (up to negligible truncation error).

---

## Implementation Notes

### Storage

You need to store $\hat{V}^\star_t(s)$ for all $t = 1, \ldots, T+1$ and all $s$ (needed for $\Delta_t$). With $T = 100$ and $|\mathcal{S}| = 1001$, this is $101 \times 1001 \approx 10^5$ floats — negligible.

Similarly store $\hat{\pi}^\star_t(s)$, $\alpha_t(s)$ for all $t$ and $s$.

You do NOT need to store $V^\star_t(s)$ for all $t$ — only $V^\star_1(s)$ (for the gap computation). But storing the optimal policies $\pi^\star_t(s)$ is needed if computing alpha_max.

### Computational cost per time step

At each $t$, you build three H-arrays ($H^{(t)}_M$, $H^{(t)}_{\hat{M}}$ or just $H^{(t)}_\Delta$, and $H^{(t)}_\alpha$), each costing $O(n \cdot (3S_{\max} + 1))$. The $\alpha$ update costs $O(|\mathcal{S}| \cdot |A(s)|)$ per state. Total over $T$ steps: roughly $T$ times the per-iteration cost of the infinite-horizon version. With $T = 100$, this is very fast.

### Reusing existing code

The H-array construction logic (`H_M`, `H_Mh`, `H_alpha`, `H_delta`) from the existing `sample_path_bound` function in `bounds.py` can be reused at each time step. The main difference is:
- Instead of iterating until convergence, you loop backward from $t = T$ to $t = 1$
- $H^{(t)}_\Delta$ is rebuilt at each $t$ using $\hat{V}^\star_{t+1}$ instead of the stationary $\hat{V}^\star$
- $\varepsilon_t(s) = \gamma^{t-1} \varepsilon(s)$ has a time-varying scale factor
- There is no $\gamma$ in front of $H^{(t)}_\alpha$ in the $\beta$ formula

### Validation

1. **Truncation check:** Verify that $|V^\star_1(s) - V^\star_\infty(s)| < 10^{-10}$ for all $s$, where $V^\star_\infty$ is the infinite-horizon value from the existing code. Same for $\hat{V}^\star_1$ and $V^{\hat{\pi}^\star}_1$.

2. **Bound validity:** Check $2\alpha_1(s) \geq V^{\hat{\pi}^\star}_1(s) - V^\star_1(s)$ for all $s$.

3. **Ordering:** $\alpha^{\max}_1(s) \leq \alpha^{\text{res}}_1(s) \leq \alpha^{\sup}_1(s)$ for all $s$.

4. **Comparison with infinite-horizon:** The finite-horizon $2\alpha_1(s)$ should be very close to the infinite-horizon $2\alpha(s)$ from the existing computation (since both are solving essentially the same problem for large $T$).

---

## Deliverables

1. **Notebook:** `notebooks/Finite_Horizon_Bounds.ipynb` containing:
   - Finite-horizon backward induction for both models
   - Policy evaluation of $\hat{\pi}^\star$ in the true model
   - Backward $\alpha$-$\beta$ recursion (all three variants)
   - Validation checks
   - Comparison plots: same format as existing (shaded regions, zoomed in/out)
   - Numerical comparison table

2. **Do not modify** any existing files (`bounds.py`, `run_new_experiments.py`, `notebooks/Sample_Path_Bounds.ipynb`, etc.). The finite-horizon analysis should be self-contained in the new notebook.
