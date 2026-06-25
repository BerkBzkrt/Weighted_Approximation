# Generate Figures for the Finite-Horizon Inventory Example

## Figure 1: Lower bound comparison (2 subfigures)

Plot all three lower bounds on the same axes, matching the style of the model approximation paper.
**Curves to plot (same colors in both subfigures):**
- $V^{\hat\pi^\star}_1(s)$: upper bound (green)
- $V^{\hat\pi^\star}_1(s) - 2\alpha^{\mathrm{res}}_1(s)$: sample-path lower bound (blue)
- $V^{\hat\pi^\star}_1(s) - C \cdot w(s)$: weighted-norm lower bound (red/orange)
- $V^{\hat\pi^\star}_1(s) - \text{sup-norm constant}$: sup-norm lower bound (another distinct color)
- Shade the region between $V^{\hat\pi^\star}_1(s)$ and the tightest (highest) lower bound, using a light fill

**Subfigure (a):** Full state space. Legend in this subfigure only.

**Subfigure (b):** Zoomed into $s \in \{-10, \ldots, 10\}$. No legend (same colors as (a)).

Use the weighted-norm and sup-norm bounds from the infinite-horizon computation (already in the codebase). This is valid since the truncation error at $T = 100$, $\gamma = 0.75$ is $< 10^{-12}$.

Save as `figures/fh_bounds_comparison.pdf`.

## Figure 2: $\alpha_{\max}$ vs $\alpha_{\mathrm{res}}$

Single plot, zoomed into $s \in \{-10, \ldots, 10\}$.

**Curves:**
- $2\alpha^{\mathrm{max}}_1(s)$ (blue solid)
- $2\alpha^{\mathrm{res}}_1(s)$ (orange solid)

Save as `figures/fh_max_vs_res.pdf`.

## Notes
- Use `plt.rcParams['pdf.fonttype'] = 42` for all figures.
- Match the existing figure style (font sizes, line widths, grid, etc.) from the codebase.
- Generate from the finite-horizon notebook data.
