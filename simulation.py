from src.dg import DataGenerator
from src.run_experiments import run
from src.utils import plot1

import torch

#torch.manual_seed(42)

#dgms = ['gaussian_mixture_5', 'gaussian_mixture_20', 'Normal', 'Logistic', 'Logistic_d20']
dgms = ['gaussian_mixture_5_d20', 'gaussian_mixture_20_d20', 'Normal_d20']

#---------- Configurations ----------
N = 2000
n_samples = 2000 # new samples
n_inits=20 # number of inits for the EM - sklearn and our method
crit = 'bic'
ct = 'diag'
pc = 0.5
pam_range = [0.05, 0.1, 0.2, 0.3, 0.4]
pc_range = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
d = 20

for dgm in dgms:
    if dgm in ['gaussian_mixture_5_d20', 'gaussian_mixture_20_d20']:
        if dgm == 'gaussian_mixture_5_d20':
            k_range = range(2, 9)
            k = 5
        else:
            k_range = range(5, 30)
            k = 20
        
        dist = 'gaussian_mixture'

        mu0 = torch.rand((k, d)) * 2 - 1 # [-1, 1]
        sigma0 = torch.rand((k, d)) * 2 # (0, 2)
        pi0 = torch.rand(k) + 0.1
        pi0 = pi0 / pi0.sum()

        dg = DataGenerator(dist, mu=mu0, sigma=sigma0, pi=pi0)


    if dgm in ['Normal_d20', 'Logistic', 'Logistic_20']:
        if dgm == 'Normal_d20': k_range = range(2, 9)
        else: k_range = range(88,95)

        if dgm == 'Logistic_20': d = 20

        dist = dgm
        alpha = 0.7
        dg = DataGenerator(dist, alpha)

    X, _ = dg.generate(N, d)

    # generate test data
    X_test = [dg.generate(N, d)[0] for _ in range(100)]

    dss, edss, _ = run(
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

    # control missingness by P(worst missing pattern)
    # dss1, edss1, _ = run(
    #     X=X, 
    #     X_test=X_test,
    #     n_samples=n_samples,
    #     k_range=k_range,
    #     crit=crit,
    #     n_inits=n_inits,
    #     ct=ct,
    #     p_range=pam_range,
    #     pc=pc, 
    #     var_complete=False
    #     )

    plot1(dss, -edss, pc_range, save_tag=dgm, legend=False)

    print(dgm, 'finish')