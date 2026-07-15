import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import os
import torch.nn.functional as F
from tqdm.auto import tqdm


#================ Optimization ================#

def opt_wb(xpi, xrho, x, sigma, block_size=1000):
    n, d = x.shape
    n_pi = xpi.shape[0]
    n_rho = xrho.shape[0]

    device = x.device

    w = torch.empty(n, d, device=device, dtype=torch.float64)
    b = torch.empty(n, 1, device=device, dtype=torch.float64)

    xpi_norm = (xpi**2).sum(dim=1, keepdim=True).T  # (1,n_pi)
    xrho_norm = (xrho**2).sum(dim=1, keepdim=True).T  # (1,n_rho)

    # solve systems blockwise to ensure memory usage linear in sample size (otherwise it might crash for large n)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        xb = x[start:end]
        bs = xb.shape[0]

        xb_norm = (xb**2).sum(dim=1, keepdim=True)

        # ---------- xpi kernel block ----------
        dist2_pi = xb_norm - 2 * xb @ xpi.T + xpi_norm
        kpix = torch.exp(-dist2_pi / (2 * sigma**2))

        kpix_mean = kpix.mean(dim=1, keepdim=True)
        Xpikpix = (kpix @ xpi) / n_pi

        # ---------- xrho kernel block ----------
        dist2_rho = xb_norm - 2 * xb @ xrho.T + xrho_norm
        krhox = torch.exp(-dist2_rho / (2 * sigma**2))

        krhox_mean = krhox.mean(dim=1, keepdim=True)
        Xrhokrhox = (krhox @ xrho) / n_rho

        # ---------- build c ----------
        c1 = Xpikpix - Xrhokrhox
        c2 = kpix_mean - krhox_mean
        c = torch.cat([c1, c2], dim=1)

        # ---------- build A ----------
        ## A11 = 1/n_rho sum_{i,j} k(x_i,xrho_j) xrho_j xrho_j^T
        A11 = torch.einsum('bk,kd,ke->bde', krhox, xrho, xrho) / n_rho
        A12 = Xrhokrhox
        A22 = krhox_mean.squeeze(1)

        A = torch.zeros(bs, d+1, d+1, dtype=torch.float64, device=device)
        A[:, :d, :d] = A11
        A[:, :d, d] = A12
        A[:, d, :d] = A12
        A[:, d, d] = A22

        # ---------- solve ----------
        # regularize to avoid (near-)singular system matrices A
        eps = 1e-5
        A += eps * torch.eye(d+1, device=A.device)
        res = torch.linalg.solve(A, c)

        w[start:end] = res[:, :d]
        b[start:end] = res[:, d:]

    return w, b


#================ Bandwidth sigma ================#

def comp_pairwise_dist(x,y):
    # in case x and y are not vectors
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    
    t1 = torch.tile(torch.sum(x**2, dim=1, keepdim=True), (1, y.shape[0]))
    t2 = -2*torch.matmul(x, y.T)
    t3 = torch.tile(torch.sum(y**2, dim=1, keepdim=True).T, (x.shape[0], 1))
    
    return t1 + t2 + t3

def sigma_heuristic_pairs(X, num_pairs=1000000):
    n = X.shape[0]

    i = torch.randint(0, n, (num_pairs,))
    j = torch.randint(0, n, (num_pairs,))

    dvals = torch.norm(X[i] - X[j], dim=1)

    return (0.5 * dvals.median()).sqrt().item()

def evaluate_sigma(xpi, xrho, sigma, splits_pi, splits_rho):
    psicon = lambda d: 0.5*d*d + d
    
    obj = 0
    
    for (train_pi, val_pi), (train_rho, val_rho) in zip(splits_pi, splits_rho):
        w, b = opt_wb(xpi[train_pi, :], xrho[train_rho, :], torch.cat([xpi[val_pi, :], xrho[val_rho, :]], dim=0), sigma)
        obj += torch.mean((w[:len(val_pi), :] * xpi[val_pi, :]).sum(dim=1) + b[:len(val_pi)].T).item()
        obj -= torch.mean(psicon((w[len(val_pi):, :] * xrho[val_rho, :]).sum(dim=1) + b[len(val_pi):].T)).item()

    return obj
    
    
#================ Main method ================#

def sample_flowgem(X0, X_obs, M, T=1000, eta=0.01, grad_tol=0.01, min_iter=10, sigma_fix=None, sigma_vals=None, cv_every=10):
    Xt = X0.clone().to(dtype = torch.float64, device = device)
    M = M.clone().to(dtype = torch.float64, device = device)
    X = X_obs.to(dtype = torch.float64, device = device)

    # determine all possible missingness patterns M
    unique_M, inverse = torch.unique(M, dim=0, return_inverse=True)

    # precompute observed indices for each m (improves computation efficiency)
    mask_idx = {tuple(m.tolist()): (m == 1).nonzero(as_tuple=True)[0] for m in unique_M}

    # samples drawn from p, i.e. conditional distribution X^(m)|M=m for different values of m
    # (dict with M as key and corresponding X^(m)'s as values)
    zpi = {m: X[inverse == i][:, mask_idx[m]].clone().to(dtype = torch.float64, device = device) for i,m in enumerate(mask_idx)}

    # prepare cross validation for sigma
    if sigma_vals is not None:
        n_jobs = min(len(sigma_vals), os.cpu_count()-1)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        splits_pi = {m: list(kf.split(zpi[m])) for m in mask_idx}
        splits_rho = list(kf.split(Xt))
    
    grads = {}
    
    # store current sigma for each pattern m
    if sigma_fix is None:
        # alternatively use heuristic (doesn't yield very good results)
        # sigma_heur = (.5*comp_pairwise_dist(Xt, Xt).flatten().median()).sqrt().item()
        sigma_heur = sigma_heuristic_pairs(Xt)
        sigma_current = {m: sigma_heur for m in mask_idx}
        print("heuristic sigma:", sigma_heur)
    else:
        sigma_current = {m: sigma_fix for m in mask_idx}
    
    mean_grad_prev = torch.inf
    
    Xhats = []
    for t in tqdm(range(T), desc="WGF iterations"):
        # print(f"Starting t = {t+1}")
        
        run_cv = sigma_vals is not None and t % cv_every == 0
        
        grad_sum = 0
        for m in mask_idx:
            if run_cv:
                # choose sigma by cross validation (see App. I.1 in MIRI paper)
                print(f"Running CV: m = {m}, size = {zpi[m].shape[0]}")
                print(f"Possible values: {sigma_vals}")
                sigma_scores = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_sigma)(zpi[m], Xt[:, mask_idx[m]], sig, splits_pi[m], splits_rho)
                    for sig in sigma_vals
                )
                sigma_current[m] = sigma_vals[np.argmax(sigma_scores)]
                print(f"Chosen sigma: {sigma_current[m]}")
                print()
                
            # optimize (w,b) according to the objective in equation (8)
            grads[m] = opt_wb(zpi[m], Xt[:, mask_idx[m]], Xt[:, mask_idx[m]], sigma_current[m])[0]
            
            # enlarge grad to R^d
            grad_full = torch.zeros_like(Xt)
            grad_full.index_copy_(1, mask_idx[m], grads[m])
            grad_sum += (zpi[m].shape[0]/X.shape[0]) * grad_full
            
        # update all particles according to WGF
        Xt = Xt + eta * grad_sum

        Xhats.append(Xt.clone().cpu().numpy())
        
        # early stopping and eta halving dependent on mean grad value
        mean_grad = (torch.mean(torch.norm(grad_sum, dim=1)) / torch.mean(torch.norm(Xt, dim=1))).item()
        if mean_grad < grad_tol and t+1 > min_iter:
            print(f"Stopped early after {t+1} iterations since mean grad {mean_grad:.4f} is below grad_tol={grad_tol}.")
            break
        if mean_grad > mean_grad_prev:
            eta = eta*0.5
            print(f"Step size eta is halved to {eta} since mean grad has increased.")
        mean_grad_prev = mean_grad

    return Xhats