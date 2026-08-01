import torch
import numpy as np

from .utils import missingness_dist, comp_density_score
from .mdm import ampute_mar
from .gaussian_mixture import GaussianMixtureMAR
from .metrics import energy_distance

from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

def run(X, X_test, n_samples, k_range, crit='bic', n_inits=20, ct='diag', p_range=None, pc=None, var_complete=True):
    _, d = X.shape
    M_full = torch.ones_like(X)
    #KDE on full data
    kde = gaussian_kde(X.T)
    kde_ed = [energy_distance(X, torch.tensor(kde.resample(n_samples)).T).item() for _ in range(100)]

    #MAR GM on full data
    gmm_full = GaussianMixtureMAR(
    k_range=k_range,
    criterion=crit,
    device='cpu',
    cov_type=ct,
    n_init=n_inits
    )
    gmm_full.fit(X, M_full)
    print('best k =', gmm_full.best_k)
    gmm_full_ed = [energy_distance(X, gmm_full.sample(n_samples)[0]) for _ in range(100)]

    #SKlearn GM
    gmm_sk = GaussianMixture(
        n_components=gmm_full.best_k,
        covariance_type='diag',
        #random_state=0,
        init_params='random',
        n_init=n_inits
    )
    gmm_sk.fit(X)
    gmm_sk_ed = [energy_distance(X, torch.tensor(gmm_sk.sample(n_samples)[0])) for _ in range(100)]

    #MAR GM on different-missingness-level data
    eds = []
    gms = []
    for p in p_range:
        # Decressing p_complete
        if var_complete:
            pp = missingness_dist(d, p)
        # Increasing p_worst_missing
        elif pc:
            pp = missingness_dist(d, pc, p)
        X0, M = ampute_mar(X, pp)
        X_filled = torch.nan_to_num(X0, nan=0.0)
        gmm = GaussianMixtureMAR(
            k_range=k_range, 
            criterion=crit,
            device='cpu', 
            cov_type=ct, 
            n_init=n_inits
        )
        gmm.fit(X_filled, M)
        print('best k =', gmm.best_k)
        gmm_ed = [energy_distance(X_filled, gmm.sample(n_samples)[0]) for _ in range(100)]
        eds.append(gmm_ed)
        gms.append(gmm)

    dss = [comp_density_score(kde, [gmm_full, *gms], x) for x in X_test]
    dss = np.array(dss)

    edss = [kde_ed, gmm_sk_ed, gmm_full_ed, *eds]

    return dss, edss