import torch

class DataGenerator:

    def __init__(self, distr="UNI", alpha=0.5, device="cpu"):
        """
        Parameters
        ----------
        distr : str
            Distribution type:
            "UNI", "Logistic", "Normal", "Student t"

        alpha : float
            Dependence parameter

        device : str
            "cpu" or "cuda"
        """

        self.distr = distr
        self.alpha = alpha
        self.device = device

    # ======================================================
    # Sample U2 conditional on U1 using copula
    # ======================================================
    def sample_x2_given_x1(self, U1):
        """
        Sample U2 | U1 from the copula.

        Conditional CDF:
        F(U2|U1)
        =
        U2 + alpha(2U1-1)(U2^2-U2)
        """
        U = torch.rand_like(U1)

        a = self.alpha * (2 * U1 - 1)
        b = 1 - a
        c = -U

        discriminant = b**2 - 4*a*c

        # numerical protection
        discriminant = torch.clamp(
            discriminant,
            min=0.0
        )

        quadratic_solution = (
            -b + torch.sqrt(discriminant)
        ) / (2*a)

        linear_solution = U / b

        U2 = torch.where(
            torch.abs(a) < 1e-10,
            linear_solution,
            quadratic_solution
        )

        return U2

    # ======================================================
    # Logistic inverse CDF
    # ======================================================
    def logistic_ppf(self, u):
        """
        Logistic quantile function:
        F^{-1}(u)=log(u/(1-u))
        """
        eps = 1e-12
        u = torch.clamp(
            u,
            eps,
            1-eps
        )
        return torch.log(u / (1-u))

    # ======================================================
    # Logistic CDF
    # ======================================================
    def logistic_cdf(self, x):
        return torch.sigmoid(x)

    # ======================================================
    # Generate complete data
    # ======================================================
    def sample_truth(self, n, d):
        if self.distr == "UNI":
            # latent uniforms
            U1 = torch.rand(
                n,
                device=self.device,
                dtype=torch.float64
            )
            U2 = self.sample_x2_given_x1(U1)
            Xrest = torch.rand(
                n,
                d-2,
                device=self.device,
                dtype=torch.float64
            )
            X = torch.column_stack([U1, U2, Xrest])

        elif self.distr == "Logistic":
            # Generate copula
            U1 = torch.rand(
                n,
                device=self.device,
                dtype=torch.float64
            )

            U2 = self.sample_x2_given_x1(U1)

            X1 = self.logistic_ppf(U1)
            X2 = self.logistic_ppf(U2)

            if d > 2:
                Xrest = self.logistic_ppf(
                    torch.rand(
                        n,
                        d-2,
                        device=self.device,
                        dtype=torch.float64
                    )
                )

                X = torch.column_stack([X1, X2, Xrest])

            else:
                X = torch.column_stack([X1, X2])

        elif self.distr == "Normal":
            mean = torch.zeros(
                d,
                device=self.device,
                dtype=torch.float64
            )

            cov = torch.eye(
                d,
                device=self.device,
                dtype=torch.float64
            )

            cov[0,1] = self.alpha
            cov[1,0] = self.alpha

            mvn = torch.distributions.MultivariateNormal(mean, cov)

            X = mvn.sample((n,))

        else:
            raise NotImplementedError(
                f"{self.distr} not implemented"
            )
        
        return X

    # ======================================================
    # Convert data back to latent uniforms
    # for missingness mechanism
    # ======================================================
    def latent_uniforms(self, X):
        if self.distr == "UNI":
            U1 = X[:,0]
            U2 = X[:,1]
        
        elif self.distr == "Logistic":
            U1 = self.logistic_cdf(X[:,0])
            U2 = self.logistic_cdf(X[:,1])

        elif self.distr == "Normal":
            normal = torch.distributions.Normal(0.0, 1.0)
            U1 = normal.cdf(X[:,0])
            U2 = normal.cdf(X[:,1])

        else:
            raise NotImplementedError

        return U1, U2

    # ======================================================
    # Generate missingness mask
    # ======================================================
    def sample_missingness(self, X):
        U1, U2 = self.latent_uniforms(X)
        P = torch.stack([(U1+U2)/3,
                         (2-U1)/3,
                         (1-U2)/3], dim=1)

        categorical = torch.distributions.Categorical(P)

        choices = categorical.sample()

        n, d = X.shape

        masks = torch.stack(
            [
                torch.ones(d, device=self.device),
                torch.arange(d, device=self.device) != 1,
                torch.arange(d, device=self.device) != 0,
            ]
        )

        M = masks[choices]

        return M

    # ======================================================
    # Main generator
    # ======================================================
    def generate(self, n, d, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        X = self.sample_truth(n, d)
        M = self.sample_missingness(X)

        return X, M