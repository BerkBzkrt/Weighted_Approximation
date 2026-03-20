import os
import numpy as np
from matplotlib import pyplot as plt


CLR_LIST = ['#1f77b4', '#ff7f0e', '#d62728', '#8c564b', '#e377c2', '#bcbd22']


def _envelope_curve(V_pi_hat_star, curves):
    """Compute the tightest (pointwise-best) bound envelope."""
    curv_arr = np.asarray(curves)
    curv_arr_shifted = V_pi_hat_star - curv_arr
    inds = np.argmin(np.abs(curv_arr_shifted - V_pi_hat_star), axis=0)
    envelope = np.zeros(len(curves[0]))
    for i in range(len(curves[0])):
        envelope[i] = curves[inds[i]][i]
    return envelope


def plot_zoomed_out(states, V_pi_hat_star, curves, labels, filename=None):
    """Zoomed-out plot showing full state range."""
    s_max = int((len(V_pi_hat_star) - 1) / 2)
    envelope = _envelope_curve(V_pi_hat_star, curves)

    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(3, 2), dpi=150)
    ax = fig.add_subplot()

    ax.plot(states, V_pi_hat_star, label=r'$V^{\hat\pi^\star}$',
            linewidth=1, color='green')
    for i, curve in enumerate(curves):
        ax.plot(states, V_pi_hat_star - curve, label=labels[i],
                linewidth=1, color=CLR_LIST[i % len(CLR_LIST)])
    ax.fill_between(states, V_pi_hat_star, V_pi_hat_star - envelope,
                     alpha=0.2, label='error band', color='red')

    plt.legend(loc='best', fontsize=5, ncol=2)
    plt.ylim(-1000, 5000)
    plt.yticks(np.arange(-1000, 5000 + 1, 1000))
    plt.xlim(-s_max, s_max)
    plt.xticks(np.arange(-s_max, s_max + 1, s_max / 2))
    plt.grid()
    plt.xlabel('state')
    plt.ylabel('value')

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_single_bound_zoomed_out(states, V_pi_hat_star, bound_curve, label,
                                  filename=None):
    """Zoomed-out plot showing V^{pi_hat*} and a single lower bound."""
    s_max = int((len(V_pi_hat_star) - 1) / 2)

    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(3, 2), dpi=150)
    ax = fig.add_subplot()

    ax.plot(states, V_pi_hat_star, label=r'$V^{\hat\pi^\star}$',
            linewidth=1, color='green')
    ax.plot(states, V_pi_hat_star - bound_curve, label=label,
            linewidth=1, color='#1f77b4')
    ax.fill_between(states, V_pi_hat_star, V_pi_hat_star - bound_curve,
                     alpha=0.2, label='error band', color='red')

    plt.legend(loc='best', fontsize=5)
    plt.ylim(-1000, 5000)
    plt.yticks(np.arange(-1000, 5000 + 1, 1000))
    plt.xlim(-s_max, s_max)
    plt.xticks(np.arange(-s_max, s_max + 1, s_max / 2))
    plt.grid()
    plt.xlabel('state')
    plt.ylabel('value')

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_single_bound_zoomed_in(states, V_pi_hat_star, bound_curve, label,
                                 s_range=(-10, 10), filename=None):
    """Zoomed-in step plot showing V^{pi_hat*} and a single lower bound."""
    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(3, 2), dpi=150)
    ax = fig.add_subplot()

    ax.step(states, V_pi_hat_star, label=r'$V^{\hat\pi^\star}$',
            linewidth=1, color='green')
    ax.step(states, V_pi_hat_star - bound_curve, label=label,
            linewidth=1, color='#1f77b4')
    ax.fill_between(states, V_pi_hat_star, V_pi_hat_star - bound_curve,
                     alpha=0.2, color='red', step='pre', label='error band')

    plt.ylim(-25, 150)
    plt.yticks(np.arange(-25, 150 + 1, 25))
    plt.xticks(np.arange(s_range[0], s_range[1] + 1, 5.0))
    plt.xlim(*s_range)
    plt.grid()
    plt.xlabel('state')
    plt.ylabel('value')

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_alpha_beta_zoomed_out(states, V_pi_hat_star, curves, labels,
                                filename=None):
    """Zoomed-out plot for alpha-beta comparison."""
    s_max = int((len(V_pi_hat_star) - 1) / 2)
    envelope = _envelope_curve(V_pi_hat_star, curves)

    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(3, 2), dpi=150)
    ax = fig.add_subplot()

    ax.plot(states, V_pi_hat_star, label=r'$V^{\hat\pi^\star}$',
            linewidth=1, color='green')
    for i, curve in enumerate(curves):
        ax.plot(states, V_pi_hat_star - curve, label=labels[i],
                linewidth=1, color=CLR_LIST[i % len(CLR_LIST)])
    ax.fill_between(states, V_pi_hat_star, V_pi_hat_star - envelope,
                     alpha=0.2, label='error band', color='red')

    plt.legend(loc='best', fontsize=5)
    plt.ylim(-1000, 5000)
    plt.yticks(np.arange(-1000, 5000 + 1, 1000))
    plt.xlim(-s_max, s_max)
    plt.xticks(np.arange(-s_max, s_max + 1, s_max / 2))
    plt.grid()
    plt.xlabel('state')
    plt.ylabel('value')

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_alpha_beta_zoomed_in(states, V_pi_hat_star, curves, labels,
                               s_range=(-10, 10), filename=None):
    """Zoomed-in step plot for alpha-beta comparison."""
    envelope = _envelope_curve(V_pi_hat_star, curves)

    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(3, 2), dpi=150)
    ax = fig.add_subplot()

    ax.step(states, V_pi_hat_star, label=r'$V^{\hat\pi^\star}$',
            linewidth=1, color='green')
    for i, curve in enumerate(curves):
        ax.step(states, V_pi_hat_star - curve, label=labels[i],
                linewidth=1, color=CLR_LIST[i % len(CLR_LIST)])
    ax.fill_between(states, V_pi_hat_star, V_pi_hat_star - envelope,
                     alpha=0.2, color='red', step='pre', label='error band')

    plt.ylim(-25, 150)
    plt.yticks(np.arange(-25, 150 + 1, 25))
    plt.xticks(np.arange(s_range[0], s_range[1] + 1, 5.0))
    plt.xlim(*s_range)
    plt.grid()
    plt.xlabel('state')
    plt.ylabel('value')

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_zoomed_in(states, V_pi_hat_star, curves, labels,
                   s_range=(-10, 10), filename=None):
    """Zoomed-in step plot around the origin."""
    envelope = _envelope_curve(V_pi_hat_star, curves)

    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(3, 2), dpi=150)
    ax = fig.add_subplot()

    ax.step(states, V_pi_hat_star, label=r'$V^{\hat\pi^\star}$',
            linewidth=1, color='green')
    for i, curve in enumerate(curves):
        ax.step(states, V_pi_hat_star - curve, label=labels[i],
                linewidth=1, color=CLR_LIST[i % len(CLR_LIST)])
    ax.fill_between(states, V_pi_hat_star, V_pi_hat_star - envelope,
                     alpha=0.2, color='red', step='pre', label='error band')

    plt.ylim(-25, 150)
    plt.yticks(np.arange(-25, 150 + 1, 25))
    plt.xticks(np.arange(s_range[0], s_range[1] + 1, 5.0))
    plt.xlim(*s_range)
    plt.grid()
    plt.xlabel('state')
    plt.ylabel('value')

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


# =========================================================================
# Phase 3: Sample-path comparison plots
# =========================================================================

def plot_shaded_comparison(states, V_pi_hat_star, V_star, sup_bound,
                           weighted_bound, sp_bound, filename=None):
    """3-panel shaded region comparison: sup-norm, weighted-norm, sample-path."""
    s_max = int((len(V_pi_hat_star) - 1) / 2)

    plt.rcParams['pdf.fonttype'] = 42
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.5), dpi=150, sharey=True)

    titles = ['Sup-norm', 'Weighted-norm', 'Sample-path']
    bounds = [sup_bound, weighted_bound, sp_bound]
    colors = ['#2ca02c', '#d62728', '#1f77b4']

    for ax, title, bound, clr in zip(axes, titles, bounds, colors):
        ax.plot(states, V_pi_hat_star, linewidth=1, color='green',
                label=r'$V^{\hat\pi^\star}$')
        ax.plot(states, V_star, linewidth=1, color='black', linestyle='--',
                label=r'$V^\star$')
        lower = V_pi_hat_star - bound
        ax.fill_between(states, V_pi_hat_star, lower,
                         alpha=0.2, color=clr)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('state', fontsize=7)
        ax.grid(True)

    axes[0].set_ylabel('value', fontsize=7)
    axes[0].legend(loc='best', fontsize=5)

    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_bound_comparison(states, gap, alpha_max, alpha_sup,
                          weighted_bound, sup_bound, filename=None):
    """Direct bound comparison on single axes, zoomed to [-10, 10]."""
    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(4, 3), dpi=150)
    ax = fig.add_subplot()

    ax.step(states, gap, linewidth=1, color='black',
            label=r'$V^{\hat\pi^\star} - V^\star$')
    ax.step(states, 2 * alpha_max, linewidth=1, color='#1f77b4',
            label=r'$2\alpha_{\max}$ (sample-path)')
    ax.step(states, 2 * alpha_sup, linewidth=1, color='#1f77b4',
            linestyle='--', label=r'$2\alpha_{\sup}$ (sample-path)')
    ax.step(states, weighted_bound, linewidth=1, color='#d62728',
            label='Weighted-norm')
    sup_val = sup_bound[0] if hasattr(sup_bound, '__len__') else sup_bound
    ax.axhline(y=sup_val, linewidth=1, color='#2ca02c',
               label='Sup-norm')

    ax.set_xlim(-10, 10)
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_xlabel('state')
    ax.set_ylabel('bound value')
    ax.legend(loc='best', fontsize=5)
    ax.grid(True)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_error_decomposition(states, epsilon, delta_at_pi_hat,
                             propagated, filename=None):
    """Error decomposition: epsilon, Delta, gamma * E[alpha] at pi_hat_star."""
    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(4, 3), dpi=150)
    ax = fig.add_subplot()

    ax.step(states, epsilon, linewidth=1, color='#1f77b4',
            label=r'$\varepsilon(s, \hat\pi^\star(s))$')
    ax.step(states, delta_at_pi_hat, linewidth=1, color='#ff7f0e',
            label=r'$\Delta(s, \hat\pi^\star(s))$')
    ax.step(states, propagated, linewidth=1, color='#2ca02c',
            label=r'$\gamma \sum_{s^\prime} \alpha(s^\prime) P(s^\prime|s,\hat\pi^\star(s))$')

    ax.set_xlim(-10, 10)
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_xlabel('state')
    ax.set_ylabel('error component')
    ax.legend(loc='best', fontsize=5)
    ax.grid(True)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_convergence(history, filename=None):
    """Semilogy plot of fixed-point convergence."""
    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(4, 3), dpi=150)
    ax = fig.add_subplot()

    ax.semilogy(range(1, len(history) + 1), history, linewidth=1,
                color='#1f77b4')
    ax.set_xlabel('iteration')
    ax.set_ylabel(r'$\max_s |\alpha^{k+1}(s) - \alpha^k(s)|$')
    ax.grid(True)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_max_vs_sup(states, alpha_max, alpha_sup, filename=None):
    """Compare 2*alpha_max vs 2*alpha_sup, zoomed to [-10, 10]."""
    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(4, 3), dpi=150)
    ax = fig.add_subplot()

    ax.step(states, 2 * alpha_max, linewidth=1, color='#1f77b4',
            label=r'$2\alpha_{\max}$')
    ax.step(states, 2 * alpha_sup, linewidth=1, color='#ff7f0e',
            label=r'$2\alpha_{\sup}$')

    ax.set_xlim(-10, 10)
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_xlabel('state')
    ax.set_ylabel('bound value')
    ax.legend(loc='best', fontsize=6)
    ax.grid(True)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()
