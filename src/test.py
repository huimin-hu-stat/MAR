import torch
import math


class TorchGaussianMixture:

    def __init__(
        self,
        n_components,
        cov_type,
        max_iter=100,
        tol=1e-4,
        reg_covar=1e-6,
        n_init=5,
        device="cpu"
    ):
        self.K = n_components
        self.cov_type = cov_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.device = device
        #self.best_params = None


    def _initialize(self, X, M):

        N, D = X.shape

        Xmiss = X.clone()
        Xmiss[M == 0] = torch.nan

        valid_rows = ~torch.isnan(Xmiss).any(dim=1)   # rows without any NaN

        if len(valid_rows) == 0:
            raise NotImplementedError('Not implemented for complete missingness -- not any datapoint completely observed')

        X_valid = Xmiss[valid_rows]
        
        # Random means
        #idx = torch.randperm(N, device=X.device)[:self.K]
        idx = torch.randperm(X_valid.shape[0])[:self.K]

        self.mu = X[idx]

        # Equal weights
        self.pi = torch.ones(
            self.K,
            device=X.device
        ) / self.K


        # Identity covariance
        self.cov = torch.ones(
            (self.K, D),
            device=X.device)


    def _estimate_log_prob(self, X, M):
        """
        Compute:
        log N(x | mu, Sigma)
        
        Diagonal covariance
        """

        N, D = X.shape

        # diff: (N,K,D)
        diff = X[:, None, :] * M[:, None, :] - self.mu[None, :, :] * M[:, None, :]

        # variance: (K,D)
        var = self.cov

        # Mahalanobis distance:
        # (N,K,D) / (K,D) -> (N,K,D)
        # 0 for missing entries
        mahal = (diff ** 2 / var[None, :, :]).sum(dim=2)

        # log determinant:
        # log |Sigma| = sum(log(var))
        # (K,D) * (N,K,D)
        # mask out the missingness
        log_det = (torch.log(var)[None, :, :] * M[:, None, :]).sum(dim=(0,2))

        return -0.5 * (
            D * math.log(2 * math.pi)
            + log_det[None, :] #(1,K)
            + mahal #(N,K)
        )


    def _e_step(self, X, M):

        log_prob = self._estimate_log_prob(X, M)

        weighted = (
            log_prob
            +
            torch.log(self.pi)
        )

        log_norm = torch.logsumexp(
            weighted,
            dim=1,
            keepdim=True
        )

        log_resp = weighted - log_norm

        return (
            log_norm.sum(),
            torch.exp(log_resp)
        )


    def _m_step(self, X, M, resp):

        N,D = X.shape

        Nk = resp.sum(dim=0)

        self.pi = Nk / N

        resp_M = resp[:, :, None] * M[:, None, :] # (N, K, D)

        Nk_M = resp_M.sum(dim=0) # (K, D)

        self.mu = (
            resp.T @ X # (K, N) @ (N, D)
        ) / Nk_M

        diff = (
            X[:,None,:]
            -
            self.mu[None,:,:]
        )

        self.cov = torch.einsum(
            "nkd,nkd,nkd->kd",
            resp_M,
            diff,
            diff
        )

        if self.cov_type == 'diag':
            self.cov /= Nk_M

        if self.cov_type == 'id':
            self.cov = self.cov.sum(dim=1)
            self.cov /= Nk_M.sum(dim=1)
            self.cov = self.cov[:, None].expand(-1, D)

        if self.cov_type == 'uni_diag':
            self.cov = self.cov.sum(dim=0)
            self.cov /= Nk_M.sum(dim=0)
            self.cov = self.cov.expand(self.K, -1)

        if self.cov_type == 'uni_id':
            self.cov = self.cov.sum()
            self.cov /= Nk_M.sum()
            self.cov = self.cov.expand(self.K, D)

        # Regularization
        self.cov = self.cov.clone()
        self.cov += self.reg_covar
        """
        eye = torch.eye(
                    D,
                    device=X.device
                )
        
        
                self.cov += (
                    self.reg_covar *
                    eye[None,:,:]
                )
        """


    def fit(self,X, M):

        X = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device
        )

        M = torch.as_tensor(
            M,
            dtype=torch.float32,
            device=self.device
        )

        best_ll = -float("inf")

        for init in range(self.n_init):

            self._initialize(X, M)

            previous = None

            for iteration in range(self.max_iter):

                ll, resp = self._e_step(X, M)

                self._m_step(
                    X,
                    M,
                    resp
                )

                if previous is not None:

                    if abs(ll-previous) < self.tol:
                        break

                previous = ll

                if ll > best_ll:

                    best_ll = ll

                    self.best_params = (
                        self.pi.clone(),
                        self.mu.clone(),
                        self.cov.clone()
                    )

        self.pi, self.mu, self.cov = self.best_params

        return self


    """
    def predict(self,X):

        X = torch.as_tensor(
            X,
            device=self.device,
            dtype=torch.float32
        )


        log_prob = (
            self._estimate_log_prob(X)
            +
            torch.log(self.pi)
        )


        return torch.argmax(
            log_prob,
            dim=1
        )
        """
    
    def sample(self, n_samples):
        # Choose mixture component for each sample
        comp = torch.multinomial(
            self.pi,
            n_samples,
            replacement=True
        )  # (N,)

        # Means and variances for selected components
        mu = self.mu[comp]          # (N, D)
        var = self.cov[comp]        # (N, D)

        # Standard normal noise
        eps = torch.randn_like(mu)

        # Sample
        X = mu + torch.sqrt(var) * eps

        return X, comp