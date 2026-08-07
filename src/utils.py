import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import os
from scipy.stats import norm, logistic, t


def missingness_dist(d, p_complete, p_all_missing=None):
    """
    Functions to set the marginal distribution of the missingness variable
    - input1: the probability of the completely observed
    - input2: the probability of the worst missingness, i.e., all missing except the last column
    """
    n_pats = 2**(d-1) - 1
    if p_all_missing is None:
        pm = np.ones(n_pats)
        pm /= pm.sum()
        pm *= 1 - p_complete
        return np.append(pm, p_complete)

    pm = np.ones(n_pats - 1)
    pm /= pm.sum()
    pm *= 1 - p_complete - p_all_missing
    pm = np.insert(pm, 0, p_all_missing)
    return np.append(pm, p_complete)


def plot(res_c, res_m, pc, pm, legend=True, is_ds=True):
    # '#ffb703'
    # Same colors used for the boxes
    if is_ds:
        colors_c = ['#8ecae6'] + ['#90be6d'] * (len(pc) + 1)
        colors_m = ['#8ecae6'] + ['#90be6d'] * (len(pm) + 1)
    else:
        colors_c = ['#fbfdff'] + ["#0a4aa9"] * (len(pc) + 1)
        colors_m = ["#fbfdff"] + ['#0a4aa9'] * (len(pm) + 1)

    labels_c = ['KDE', 1, *pc]
    labels_m = ['KDE', 0, *pm]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax in axes:
        ax.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

    # Legend
    if is_ds:
        legend_handles = [
            Patch(facecolor='#8ecae6', edgecolor='black', alpha=0.7, label='Gaussian KDE'),
            Patch(facecolor='#90be6d', edgecolor='black', alpha=0.7, label='MAR Gaussian Mixture')
        ]

    else:
        legend_handles = [
                    Patch(facecolor="#fbfdff", edgecolor='black', alpha=0.7, label='Gaussian KDE'),
                    Patch(facecolor='#0a4aa9', edgecolor='black', alpha=0.7, label='MAR Gaussian Mixture')
                ]
    

    # left plot
    bp1 = axes[0].boxplot(
        res_c,
        patch_artist=True,
        showfliers=False,
        widths=0.3
    )
    axes[0].set_xlabel(r'$\mathbb{P}(M=\mathbf{0})$', fontsize=13)
    axes[0].set_xticks(
        range(1, len(labels_c) + 1),
        labels_c,
        fontsize=11,
        rotation=0
    )

    for patch, color in zip(bp1['boxes'], colors_c):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # right plot
    bp2 = axes[1].boxplot(
        res_m,
        patch_artist=True,
        showfliers=False,
        widths=0.2
    )
    axes[1].set_xlabel(r'$\mathbb{P}(M=(1,1,0))$', fontsize=13)
    axes[1].set_xticks(
        range(1, len(labels_m) + 1),
        labels_m,
        fontsize=11,
        rotation=0
    )
    
    for patch, color in zip(bp2['boxes'], colors_m):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    if legend:
        # shared legend
        fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),
        frameon=True,
        fontsize=11,
        ncol=1
        )

    # Shared y label
    if is_ds:
        fig.text(
            -0.01,              # horizontal position
            0.5,               # vertical position
            "Density score",
            va="center",
            rotation="vertical",
            fontsize=13
        )
    else:
        fig.text(
            -0.01,              # horizontal position
            0.5,               # vertical position
            "Energy distance",
            va="center",
            rotation="vertical",
            fontsize=13
        )

    plt.tight_layout()
    plt.show()


def plot1(ds, ned, pc, save_to_path=None, legend=False):

    os.makedirs("results/figs", exist_ok=True)

    # '#ffb703'
    # Same colors used for the boxes
    colors = ['#8ecae6'] + ['#90be6d'] * (len(pc) + 1)

    labels = ['KDE', 1, *pc]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax in axes:
        ax.grid(
            True,
            linestyle='--',
            alpha=0.5
        )


    legend_handles = [
        Patch(facecolor='#8ecae6', edgecolor='black', alpha=0.7, label='Gaussian KDE'),
        Patch(facecolor='#90be6d', edgecolor='black', alpha=0.7, label='MAR Gaussian Mixture')
    ]

    # left plot
    bp1 = axes[0].boxplot(
        ds,
        patch_artist=True,
        showfliers=False,
        widths=0.3
    )
    #axes[0].set_xlabel(r'$\mathbb{P}(M=\mathbf{0})$', fontsize=13)
    axes[0].set_xticks(
        range(1, len(labels) + 1),
        labels,
        fontsize=11,
        rotation=0
    )
    axes[0].set_title('Density score')

    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # right plot
    bp2 = axes[1].boxplot(
        ned,
        patch_artist=True,
        showfliers=False,
        widths=0.2
    )
    #axes[1].set_xlabel(r'$\mathbb{P}(M=(1,1,0))$', fontsize=13)
    axes[1].set_xticks(
        range(1, len(labels) + 1),
        labels,
        fontsize=11,
        rotation=0
    )
    axes[1].set_title('Negative energy distance')
    
    for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    if legend:
        # shared legend
        fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),
        frameon=True,
        fontsize=11,
        ncol=1
        )

    fig.text(
        0.5,              # horizontal position
        0,               # vertical position
        r'$\mathbb{P}(M=\mathbf{0})$',
        va="center",
        fontsize=13
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join('results/figs', save_to_path),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# functions to numerically find the quantile of a 1D Gaussian mixture
def gmm_cdf(x, mu, sigma, pi):
    z = (x - mu) / sigma
    Phi = 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2., device=z.device))))
    return (pi * Phi).sum()

def gmm_quantile(p, mu, sigma, pi, tol=1e-6, max_iter=1000):
    """
    Compute x such that P(X <= x) = p for a 1D Gaussian mixture.
    use binary search (bisection) to numerically find x.
    """
    lo = (mu - 10 * sigma).min()
    hi = (mu + 10 * sigma).max()

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if gmm_cdf(mid, mu, sigma, pi) < p:
            lo = mid
        else:
            hi = mid

        if (hi - lo) < tol:
            break

    return (lo + hi) / 2

def true_quantile(dist, p=0.1, mu=None, sigma=None, pi=None, tol=1e-6, max_iter=100, df=None):
    if dist == 'Normal':
        return norm.ppf(p)
    elif dist == 'gaussian_mixture' and all(x is not None for x in (mu, sigma, pi)):
        return gmm_quantile(p, mu, sigma, pi, tol, max_iter)
    elif dist == 'Logistic':
        return logistic.ppf(p)
    elif dist == 'Student t':
        return t.ppf(p, df)