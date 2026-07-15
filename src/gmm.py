import torch
import numpy as np

class GMM:
    def __init__(self, X, M, cov_type, eps=1e-12):
        """
        X: (N, d)
        M: (N, d) binary mask
        cov_type: diag, id, uni_diag, uni_id
        """
        self.X = X.float() # ensure float32 consistency
        self.dtype, self.device = self.X.dtype, self.X.device
        self.M = M.to(dtype=self.dtype, device=self.device)

        self.N, self.d = X.shape
        self.cov_type = cov_type
        self.eps = eps
    
    def log_gaussian_masked(self, mu, sigma):
        """Compute logp(xi) for each cluster
        """
        mu = mu.to(dtype=self.dtype, device=self.device) # (K, d)
        sigma = sigma.to(dtype=self.dtype, device=self.device) # (K, d)

        K, _ = mu.shape

        # --- correct expansion ---
        X = self.X.unsqueeze(0).expand(K, self.N, self.d) # replicate X along axis 0
        M = self.M.unsqueeze(0).expand(K, self.N, self.d)

        mu = mu[:, None, :]          # (K,1,d)
        var = (sigma ** 2)[:, None, :]

        log_prob = -0.5 * (
            ((X - mu) ** 2) / var
            + torch.log(2 * torch.pi * var)
        )

        # the log_prob of the missing entries are 0
        log_prob = log_prob * M

        return log_prob.sum(dim=2)  # (K,N)
    
    def observed_loglik(self, mu, sigma, pi):
        """
        Observed loglikelihood: responsibility weighted
        """
        log_px = self.log_gaussian_masked(mu, sigma)  # (K,N)

        log_mix = (
            torch.log(pi + self.eps).unsqueeze(1)
            + log_px
        )
    
        return torch.logsumexp(log_mix, dim=0).sum()
    
    def em1(self, K, max_iters=1000, mu=None, sigma=None, pi=None):
        """
        EM for fixed K -- number of components
        """
        K = int(K)
        # ---- init ----
        if mu is None:
            idx = torch.randint(0, self.N, (K,), device=self.device)
            mu = self.X[idx] + 0.1 * torch.randn(K, self.d, device=self.device)
        else:
            mu = mu.to(device=self.device, dtype=self.dtype)

        if sigma is None:
            sigma = torch.ones(K, self.d, device=self.device)
        else:
            sigma = sigma.to(device=self.device, dtype=self.dtype)

        if pi is None:
            pi = torch.ones(K, device=self.device) / K
        else:
            pi = pi.to(device=self.device, dtype=self.dtype)

        it=0

        for _ in range(max_iters):

            it=it+1
            
            loglik_old = self.observed_loglik(mu, sigma, pi)

            # =====================
            # E-step
            # =====================
            log_px = self.log_gaussian_masked(mu, sigma)  # (K, N)

            log_pi = torch.log(pi + self.eps).unsqueeze(1)      # (K, 1)

            log_r = log_pi + log_px
            log_r = log_r - torch.logsumexp(log_r, dim=0, keepdim=True)

            r = torch.exp(log_r)  # (K, N)

            # =====================
            # M-step
            # =====================
            Nk = r.sum(dim=1, keepdim=True) + self.eps  # (K, 1)

            pi = (Nk[:, 0] / self.N)

            # --- mean update (masked properly) ---
            Xm = self.X * self.M  # (N, d)

            mu = (r @ Xm) / (r @ self.M + self.eps)

            # =====================
            # variance update
            # =====================
            diff = self.X.unsqueeze(0) - mu.unsqueeze(1)  # (K, N, d)

            if self.cov_type == 'diag':
                sigma = torch.sqrt(
                (r.unsqueeze(2) * (diff ** 2) * self.M.unsqueeze(0)).sum(dim=1)
                / (r @ self.M + self.eps)
                + self.eps
                )

            elif self.cov_type == 'id':
                sigma = torch.sqrt(
                (r * ((diff ** 2) * self.M.unsqueeze(0)).sum(dim=2)).sum(dim=1)
                / ((r.unsqueeze(2) * self.M.unsqueeze(0)).sum(dim=(1,2)) + self.eps)
                + self.eps
                ).repeat(self.d, 1).T

            elif self.cov_type == 'uni_diag':
                sigma = torch.sqrt(
                    (r.unsqueeze(2) * (diff ** 2) * self.M.unsqueeze(0)).sum(dim=(0,1))
                    / ((r.unsqueeze(2) * self.M.unsqueeze(0)).sum(dim=(0,1)) + self.eps)
                    + self.eps
                ).repeat(K, 1) # d    

            elif self.cov_type == 'uni_id':
                sigma = torch.sqrt(
                    (r * (((diff ** 2) * self.M.unsqueeze(0)).sum(dim=2))).sum()
                    / ((r.unsqueeze(2) * self.M.unsqueeze(0)).sum() + self.eps)
                    + self.eps
                ).repeat(K, self.d)

            loglik_new = self.observed_loglik(mu, sigma, pi)
            rel_change = torch.abs(loglik_new - loglik_old) / (torch.abs(loglik_old) + self.eps)
            if rel_change <= 0:
                print(f"Converged at iteration {it}, rel_change={rel_change.item():.2e}")
            break

        return mu, sigma, pi
    
    def _num_p(self, K):
        if self.cov_type == 'diag':
            return (2 * self.d + 1) * K - 1
        elif self.cov_type == 'id':
            return (self.d + 2) * K - 1
        elif self.cov_type == 'uni_diag':
            return (self.d + 1) * K + self.d - 1
        elif self.cov_type == 'uni_id':
            return (self.d + 1) * K
    
    def aic(self, loglik, p):
        return -2 * loglik + 2 * p
    
    def bic(self, loglik, p):
        return -2 * loglik + p * np.log(self.N)
    
    def sample_gmm(self, mu, sigma, pi, n_samples):
        """
        mu: (K, d)
        sigma: (K, d)
        pi: (K,)
        n_samples: int

        returns: (n_samples, d)
        """
        # normalize weights
        pi = torch.softmax(pi, dim=0)

        # 1. sample component indices
        z = torch.multinomial(pi, n_samples, replacement=True)  # (N,)

        # 2. sample standard normals
        eps = torch.randn(n_samples, self.d)

        # 3. gather parameters for each sample
        mu_z = mu[z]        # (N, d)
        sigma_z = sigma[z]  # (N, d)

        # 4. reparameterization
        x = mu_z + sigma_z * eps

        return x
    
    def iterEM(self, n_k, start, step, n_mc, n_samples):
        kk = [x * step + start for x in range(1, n_k + 1)]
        q = np.empty((n_k, n_mc))
        loglik = np.empty(n_k)
        aic = np.empty(n_k)
        bic = np.empty(n_k)

        muhat = None
        sigmahat = None
        pihat = None

        for i in range(n_k):
            if i == 0:
                mu=None
                sigma=None
                pi= None
            else:
                pi=torch.cat([pi, torch.zeros(step, device=pi.device, dtype=pi.dtype)])
                mu = torch.cat([mu, torch.zeros(step, mu.shape[1], device=mu.device, dtype=mu.dtype)], dim=0)
                sigma = torch.cat([sigma, sigma[0:1].repeat(step, 1)], dim=0)

            mu, sigma, pi = self.em1(kk[i], mu=mu, sigma=sigma, pi=pi)

            q[i]= [np.quantile(self.sample_gmm(mu, sigma, pi, n_samples), 0.1) for _ in range(n_mc)]
            loglik[i] = self.observed_loglik(mu, sigma, pi)
            p = self._num_p(kk[i])
            aic[i] = self.aic(loglik[i], p)
            bic[i] = self.bic(loglik[i], p)

            if i == 0:
                muhat = mu
                sigmahat = sigma
                pihat = pi
            elif aic[i] <= np.min(aic[:i]):
                muhat = mu
                sigmahat = sigma
                pihat = pi
            print(f"{i} {kk[i]:.2f} {loglik[i]:.2f} {aic[i]:.2f} {bic[i]:.2f}")

        return muhat, sigmahat, pihat, kk, q, loglik, aic, bic
