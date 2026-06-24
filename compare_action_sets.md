# Compare Two Restricted Action Set Formulations

## Task

We have two candidate formulations for the restricted action set at each state $s$:

**Set 1 (current):**
$$A_1(s) = \{\max(0, \sigma - s) : \sigma \in \{0, \ldots, n\}\} \cup \{\hat\pi^\star_t(s)\}$$

**Set 2 (simpler):**
$$A_2(s) = \{0, 1, \ldots, \max(0, n - s)\} \cup \{\hat\pi^\star_t(s)\}$$

These are identical for $s \geq 0$ but differ for $s < 0$:
- Set 1 at $s < 0$: $\{|s|, |s|+1, \ldots, n+|s|\}$ (actions corresponding to base-stock levels $0, \ldots, n$)
- Set 2 at $s < 0$: $\{0, 1, \ldots, n+|s|\}$ (includes extra small actions $0, \ldots, |s|-1$ that don't correspond to any base-stock level in $\{0, \ldots, n\}$)

Set 2 is a superset of Set 1 for $s < 0$, so $\alpha$ computed with Set 2 should satisfy $\alpha_{\text{Set 1}}(s) \leq \alpha_{\text{Set 2}}(s)$.

## What to do

1. Implement `alpha_restricted` using Set 2 (in addition to the existing Set 1 implementation).
2. Run both on the finite-horizon inventory example ($T = 100$, $\gamma = 0.75$).
3. Compare $2\alpha_1(s)$ from both formulations — plot them together, zoomed into $s \in \{-10, \ldots, 10\}$.
4. Report $\max_s |\alpha_{\text{Set 1}}(s) - \alpha_{\text{Set 2}}(s)|$ and whether the difference is negligible or meaningful.

This is a quick check — no need for a separate notebook, just add a cell to the existing finite-horizon notebook.
