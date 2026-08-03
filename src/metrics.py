#%Hellinger distance, energy distance, quantile of GMM

import torch
from torch.distributions import Normal, Independent, MultivariateNormal
from scipy.spatial.distance import cdist

# def energy_distance(X, Y):
#     """
#     Compute the (squared) energy distance between two samples X ~ P, Y ~ Q.
#     X: tensor of shape (n, d)
#     Y: tensor of shape (m, d)
    
#     E(P, Q) = 2 * E||X - Y|| - E||X - X'|| - E||Y - Y'||
#     """
#     X = X.double()
#     Y = Y.double()

#     n, m = X.shape[0], Y.shape[0]

#     # Pairwise distances
#     d_XY = torch.cdist(X, Y, p=2)      # (n, m)
#     d_XX = torch.cdist(X, X, p=2)      # (n, n)
#     d_YY = torch.cdist(Y, Y, p=2)      # (m, m)

#     term_XY = d_XY.mean()

#     # exclude diagonal (i.e. X_i vs X_i, which is 0 and shouldn't bias things,
#     # but standard practice is to use the unbiased version excluding self-pairs)
#     term_XX = (d_XX.sum() - d_XX.diagonal().sum()) / (n * (n - 1))
#     term_YY = (d_YY.sum() - d_YY.diagonal().sum()) / (m * (m - 1))

#     ed = 2 * term_XY - term_XX - term_YY

#     return ed #torch.clamp(ed, min=0.0) # avoid negative values

def energy_distance(X, Y):
    XY = cdist(X, Y)
    XX = cdist(X, X)
    YY = cdist(Y, Y)
    return (2 * XY.mean() - XX.mean() - YY.mean())* X.shape[0] / 2

# Example usage:
# data: tensor of shape (n, 3)
# samples: sample_gmm(mu, sigma, pi, param_vals['n_new'])  -> tensor of shape (m, 3)


def hellinger_normal_gmm(mu0, sigma0, mu, sigma, pi, n_points=10000):
    """
    Compute the hellinger distance of a univariate gaussian density with the joint density of Gaussian mixture
    """
    lo = min(mu0 - 5*sigma0, (mu - 5*sigma).min())
    hi = max(mu0 + 5*sigma0, (mu + 5*sigma).max())

    x = torch.linspace(lo, hi, n_points)

    dx = x[1] - x[0]

    # Normal density
    p = Normal(mu0, sigma0).log_prob(x).exp()

    # GMM density
    q = sum(
        pi[k] * Normal(mu[k], sigma[k]).log_prob(x).exp()
        for k in range(len(pi))
    )

    bc = torch.sum(torch.sqrt(p * q)) * dx

    return torch.sqrt(1 - bc)


def hellinger_gaussian_diag_gmm(
    mu0,
    cov0,
    mu,
    sigma,
    pi,
    n_samples=100_000,
):
    """
    Hellinger distance between

        N(mu0, diag(sigma0^2))

    and

        Σ_k pi_k N(mu[k], diag(sigma[k]^2))
    """
    pi = pi / pi.sum()
    # Reference Gaussian
    p = MultivariateNormal(mu0, cov0)
    # Sample from p
    x = p.sample((n_samples,))              # (N, d)
    log_p = p.log_prob(x)                   # (N,)

    # Log-density of each GMM component
    components = Independent(
        Normal(mu, sigma),
        1,
    )

    # (N, K)
    log_comp = components.log_prob(x[:, None, :])
    # Mixture log-density
    log_q = torch.logsumexp(
        torch.log(pi) + log_comp,
        dim=1,
    )
    # Bhattacharyya coefficient
    bc = torch.exp(0.5 * (log_q - log_p)).mean()
    return torch.sqrt(torch.clamp(1 - bc, min=0.0))


def hellinger_distance(p1, p2, eps=1e-8):
    """
    Compute Hellinger distance between two probability vectors.

    Args:
        p1: tensor of probabilities, shape (..., K)
        p2: tensor of probabilities, shape (..., K)

    Returns:
        Hellinger distance, shape (...)
    """
    p1 = p1 / (p1.sum(dim=-1, keepdim=True) + eps)
    p2 = p2 / (p2.sum(dim=-1, keepdim=True) + eps)

    return torch.sqrt(
        torch.sum((torch.sqrt(p1) - torch.sqrt(p2)) ** 2, dim=-1)
    ) / torch.sqrt(torch.tensor(2.0, device=p1.device))


