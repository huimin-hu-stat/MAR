from src.dg import DataGenerator
from src.run_experiments import run
from src.utils import plot1

import torch
import numpy as np

torch.manual_seed(42)

# global parameters
N = 2000

# new samples
n_samples = 2000

# number of inits for the EM - sklearn and our method
n_inits=20

crit = 'bic'
ct = 'diag'

pc = 0.5
pam_range = [0.05, 0.1, 0.2, 0.3, 0.4]

pc_range = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


d = 3

# Gaussina Mixture
# k_range = range(5, 30)
# dist = 'gaussian_mixture'
# # true parameters of the gaussian mixture
# k = 20
# mu0 = torch.rand((k, d)) * 2 - 1 # [-1, 1]
# sigma0 = torch.rand((k, d)) * 2 # (0, 2)
# pi0 = torch.rand(k) + 0.1
# pi0 = pi0 / pi0.sum()
# dg = DataGenerator(dist, mu=mu0, sigma=sigma0, pi=pi0)


# Logistic
# k_range = range(88,95)
# dist = 'Logistic'

k_range = range(2, 20)
dist = 'Normal'
alpha = 0.7
dg = DataGenerator(dist, alpha)

X, _ = dg.generate(N, d)

# generate test data
X_test = [dg.generate(N, d)[0] for _ in range(100)]

dss, edss = run(
    X=X, 
    X_test=X_test,
    n_samples=n_samples,
    k_range=k_range,
    crit=crit,
    n_inits=n_inits,
    ct=ct,
    p_range=pc_range,
    pc=None, 
    var_complete=True
    )

dss1, edss1 = run(
    X=X, 
    X_test=X_test,
    n_samples=n_samples,
    k_range=k_range,
    crit=crit,
    n_inits=n_inits,
    ct=ct,
    p_range=pam_range,
    pc=pc, 
    var_complete=False
    )

edss1 = np.array(edss1)
edss1_ = np.delete(edss1, 1, axis=0).T

edss = np.array(edss)
edss_ = np.delete(edss, 1, axis=0).T

plot1(dss, -edss_, pc_range, save_tag='Normal_ds_ed.png', legend=False)