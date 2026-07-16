import torch
from typing import Optional, Tuple

from torch.distributions import Normal, StudentT, MultivariateNormal, Categorical


# TO DO: add gaussian mixture data

class DataGenerator:

    def __init__(
        self,
        #d: int = 3,
        #N: int = 2000,
        distr: str = "Normal",
        mu: torch.tensor = None,
        sigma: torch.tensor = None,
        pi: torch.tensor = None,
        alpha: float = 0.7,
        df: Optional[int] = None,
        device: str = "cpu",
        dtype=torch.float64,
    ):
        #self.d = d
        #self.N = N
        self.distr = distr
        self.alpha = alpha
        self.df = df
        self.device = device
        self.dtype = dtype

        self.mu = mu
        self.sigma = sigma
        self.pi = pi

        if distr == "Student t" and df is None:
            raise ValueError(
                "Student t requires df"
            )
        
        if distr == 'gaussian_mixture' and (mu is None or sigma is None or pi is None):
            raise ValueError(
                "Gassian mixture requires mu, sigma, pi"
            )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def rand(self, *shape):

        return torch.rand(
            *shape,
            device=self.device,
            dtype=self.dtype
        )


    # --------------------------------------------------
    # Copula
    # --------------------------------------------------
    def sample_copula(
        self,
        n: int,
        d: int
    ):
        """
        Generate dependent uniforms.
            conditional CDF: F(x2|x1) = x2 + alpha*(2*x1-1)*(x2^2 - x2)
            invert this to sample X2 = F_inv(U)
            Inversion yields quadratic equation: alpha*(2*x1-1)*x2^2 + [1 - alpha*(2*x1-1)]*x2 - u = 0
            NOTE: CDF is symmetric so the same method can be used to sample X1 given X2
        """
        U1 = self.rand(n)
        a = self.alpha * (2 * U1 - 1)
        b = 1 - a
        disc = b ** 2 + 4 * a * U1
        U2 = (-b + torch.sqrt(disc)) / (2*a)

        # linear case when a≈0
        U2 = torch.where(torch.abs(a)<1e-10, U1 / b, U2)
        U2 = torch.clamp(U2, 0, 1)

        Urest = self.rand(n, d-2)

        return torch.column_stack([U1, U2, Urest])
    

    # --------------------------------------------------
    # Marginal inverse CDF
    # --------------------------------------------------
    def inverse_cdf(
        self,
        U
    ):
        if self.distr == "UNI":
            return U

        elif self.distr == "Logistic":
            eps = torch.finfo(
                U.dtype).eps
            U = torch.clamp(
                U,
                eps,
                1-eps)
            return torch.log(U/(1-U))

        elif self.distr == "Normal":
            normal = Normal(0, 1)
            return normal.icdf(U)

        elif self.distr == "Student t":
            tdist = StudentT(self.df)
            return tdist.icdf(U)

        else:
            raise NotImplementedError

    # --------------------------------------------------
    # Gaussian special case
    # --------------------------------------------------
    def sample_gaussian(
        self,
        n,
        d
    ):
        mean = torch.zeros(
            d,
            device=self.device,
            dtype=self.dtype
        )

        cov = torch.eye(
            d,
            device=self.device,
            dtype=self.dtype
        )

        cov[0,1] = self.alpha
        cov[1,0] = self.alpha

        mvn = MultivariateNormal(mean, cov)

        return mvn.sample((n,))
    
    def sample_gmm(
        self,
        n
    ):
        _, d = self.mu.shape

        # normalize weights
        pi = torch.softmax(self.pi, dim=0)

        # 1. sample component indices
        z = torch.multinomial(pi, n, replacement=True)  # (N,)

        # 2. sample standard normals
        eps = torch.randn(n, d)

        # 3. gather parameters for each sample
        mu_z = self.mu[z]        # (N, d)
        sigma_z = self.sigma[z]  # (N, d)

        # 4. reparameterization
        x = mu_z + sigma_z * eps
        return x
    

    # --------------------------------------------------
    # Truth generation
    # --------------------------------------------------
    def sample_truth(
        self,
        n:int,
        d:int
    ):
        if self.distr == "Normal":
            return self.sample_gaussian(n, d)
        
        if self.distr == 'gaussian_mixture':
            return self.sample_gmm(n)

        U = self.sample_copula(n, d)

        return self.inverse_cdf(U)

    # --------------------------------------------------
    # Latent uniforms
    # --------------------------------------------------
    def latent_uniforms(self, X):
        X12 = X[:,:2]
        if self.distr == "UNI":
            return X12[:,0], X12[:,1]

        if self.distr == "Logistic":
            return (
                torch.sigmoid(X12[:,0]),
                torch.sigmoid(X12[:,1])
            )

        if self.distr == "Normal":
            normal = Normal(0,1)
            return (
                normal.cdf(X12[:,0]),
                normal.cdf(X12[:,1])
            )

        if self.distr == "Student t":
            tdist = StudentT(self.df)
            return (
                tdist.cdf(X12[:,0]),
                tdist.cdf(X12[:,1])
            )
        
        if self.distr == 'gaussian_mixture':
            
            normal1 = Normal(self.mu[:, 0].unsqueeze(1), self.sigma[:, 0].unsqueeze(1))
            normal2 = Normal(self.mu[:, 1].unsqueeze(1), self.sigma[:, 1].unsqueeze(1))

            return (
                self.pi @ normal1.cdf(X12[:,0].repeat(len(self.mu), 1)),
                self.pi @ normal2.cdf(X12[:,1].repeat(len(self.mu), 1))
            )

        raise NotImplementedError

    # --------------------------------------------------
    # Missingness
    # --------------------------------------------------
    def sample_missingness(
        self,
        X
    ):
        U1,U2 = self.latent_uniforms(X)
        probs = torch.stack(
            [
                (U1+U2)/3,
                (2-U1)/3,
                (1-U2)/3
            ],
            dim=1
        )

        choice = Categorical(probs).sample()

        n,d = X.shape

        masks = torch.stack(
            [
                torch.ones(
                    d,
                    device=self.device
                ),

                torch.arange(
                    d,
                    device=self.device
                ) != 1,

                torch.arange(
                    d,
                    device=self.device
                ) != 0,
            ]
        )

        return masks[choice]


    # --------------------------------------------------
    # Main
    # --------------------------------------------------
    def generate(
        self,
        n:int,
        d:int,
        seed:Optional[int]=None
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
        X = self.sample_truth(n, d)
        M = self.sample_missingness(X)
        return X,M