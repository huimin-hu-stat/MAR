import torch
import numpy as np

from .utils import missingness_dist
from .mdm import ampute_mar
from .gaussian_mixture import GaussianMixtureMAR
from .metrics import energy_distance

from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture


def comp_density_score(kde, gms, X):
    kde_score = np.mean(kde.logpdf(X.T))
    gms_score = np.array([gm.evaluate(X).item() for gm in gms])
    return np.insert(gms_score, 0, kde_score)

def run(
        X, # train
        X_test, # test
        n_samples, # sample size for test
        k_range, 
        crit='bic', # selection criterion of k
        n_inits=20, # multi initiations
        ct='diag', # diag, id, uni_diag, uni_id
        p_range=None, # 1) prob(fully observed); 2) prob(worst missingness)
        pc=None, # fixed prob(fully observed)
        var_complete=True # if varying prob(fully observed); 2) else varying prob(worst missingness)
        ):
    
    _, d = X.shape
    m = len(X_test)

    # Mask for complete data
    M_full = torch.ones_like(X)

    #------ KDE on complete data
    kde = gaussian_kde(X.T)
    kde_ed = [energy_distance(X_test[i], torch.tensor(kde.resample(n_samples)).T).item() for i in range(m)]

    #------ MAR GM on complete data
    gmm_full = GaussianMixtureMAR(
    k_range=k_range,
    criterion=crit,
    device='cpu',
    cov_type=ct,
    n_init=n_inits
    )
    # fit on complete data
    gmm_full.fit(X, M_full)
    print('best k =', gmm_full.best_k)
    # energy distance gainst train X
    gmm_full_ed = [energy_distance(X_test[i], gmm_full.sample(n_samples)[0]) for i in range(m)]

    #------- SKlearn GM
    gmm_sk = GaussianMixture(
        n_components=gmm_full.best_k,
        covariance_type='diag',
        #random_state=0,
        init_params='random',
        n_init=n_inits
    )
    # fit on complete data
    gmm_sk.fit(X)
    # energy distance gainst train X
    gmm_sk_ed = [energy_distance(X_test[i], torch.tensor(gmm_sk.sample(n_samples)[0])) for i in range(m)]

    #-------- MAR GM on different-missingness-level data
    eds = []
    gms = []
    for p in p_range:
        # varying prob(fully observed)
        if var_complete:
            pp = missingness_dist(d, p)
        # varying prob(worst missingness)
        elif pc:
            pp = missingness_dist(d, pc, p)
        # induce missingness
        X0, M = ampute_mar(X, pp)
        X_filled = torch.nan_to_num(X0, nan=0.0)
        gmm = GaussianMixtureMAR(
            k_range=k_range, 
            criterion=crit,
            device='cpu', 
            cov_type=ct, 
            n_init=n_inits
        )
        # fit on missing data
        gmm.fit(X_filled, M)
        print('best k =', gmm.best_k)
        # energy distance gainst train X
        gmm_ed = [energy_distance(X_test[i], gmm.sample(n_samples)[0]) for i in range(m)]
        eds.append(gmm_ed)
        gms.append(gmm)

    dss = np.array(
        [comp_density_score(kde, [gmm_full, *gms], x) for x in X_test]
    )

    edss = np.array(
        [kde_ed, gmm_full_ed, *eds]
    ).T

    edss_sk = np.array(
        [kde_ed, gmm_sk_ed, gmm_full_ed, *eds]
    ).T

    return dss, edss, edss_sk