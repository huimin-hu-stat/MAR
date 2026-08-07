# ============================================================
# DP Gaussian Mixture with Shared Covariance
# Exact implementation of Algorithms 1 and 2
# ============================================================

library(mvtnorm)
library(MCMCpack)


# Helper function: sample from inverse-Wishart
rinvwishart <- function(df, scale_matrix) {
  # Sample from Wishart and invert
  wishart_sample <- rWishart(1, df, solve(scale_matrix))[,,1]
  return(solve(wishart_sample))
}

# Helper function: compute log-likelihood for multivariate normal
dmvnorm_log <- function(x, mu, Sigma) {
  Rfast::dmvnorm(x, mu = mu, sigma = Sigma, log = TRUE)
}


loglikGauss<-function(data, Sigma, mu){
  
  
  M<- is.na(data) * 1
  patterns<-unique(M)
  
  n <- nrow(data)
  d <- length(mu)
  
  mu<-matrix(mu, nrow=d, ncol=1)
  
  log_density<-0
  
  # Second term: Pattern-specific likelihood terms
  for (j in 1:nrow(patterns)) {
    pattern<-patterns[j,]
    pattern_rows <- which(rowSums(unname(M==matrix(pattern, nrow = nrow(M), ncol = d, byrow = TRUE)))==d)
    nS<-length(pattern_rows)
    index<-1:d
    observedindex<-index[pattern==0]
    
    X_m<-as.matrix(data[pattern_rows,observedindex, drop=F])
    n_m<-nrow(X_m)
    
    if (n_m > 0) {
      # Compute Sigma_m (could be pattern-specific transformation of Sigma)
      Sigma_m <- Sigma[observedindex,observedindex]
      mu_m <-mu[observedindex,]
      
    
      
      log_density<-log_density+ sum(Rfast::dmvnorm(x = X_m, mu=mu_m, sigma = as.matrix(Sigma_m), log = T))
      
      # # Compute sum over observations in pattern m
      # centered <- X_m - matrix(mu_m, n_m, d, byrow = TRUE)
      # quad_form <- sum(diag(centered %*% solve(Sigma_m) %*% t(centered)))
      # log_density <- log_density - (n_m/2) * quad_form
      # 
      # # Add determinant term
      # log_density <- log_density - (n_m/2) * log(det(Sigma_m))
      # 
      # # Scatter matrix S_m
      # X_m_bar <- colMeans(X_m)
      # S_m <- t(X_m - matrix(X_m_bar, n_m, d, byrow = TRUE)) %*%
      #   (X_m - matrix(X_m_bar, n_m, d, byrow = TRUE))
      # log_density <- log_density - (1/2) * sum(diag(solve(Sigma_m) %*% S_m))
      
      
    }
    
    
  }
  
  return(log_density)
  
}



# ============================================================
# Algorithm 2: UpdateAssignmentsShared
# ============================================================

UpdateAssignmentsShared <- function(X, z, K, mu_mat, Sigma, alpha,
                                    mu0, tau0_sq) {
  # Inputs:
  # X       : n x d data matrix
  # z       : n-vector of cluster assignments
  # K       : current number of clusters
  # mu_mat  : K x d matrix of cluster centers
  # Sigma   : d x d shared covariance matrix
  # alpha   : DP concentration parameter
  # mu0     : d-vector prior mean for centers
  # tau0_sq : scalar prior variance for new centers
  
  n <- nrow(X)
  d <- ncol(X)
  
  M<- is.na(X) * 1
  patterns<-unique(M)
  
  n <- nrow(X)
  d <- length(mu0)
  
  #mu<-matrix(mu, nrow=d, ncol=1)
  
  log_density<-0
  
  Sigma_chol<-list()
  log_det_Sigma<-list()
  const_term<-list()
  Sigma_prop_chol<-list()
  
    # Pre-compute proposal covariance for new centers (Line 13)
    # mu_new ~ N(mu0, Sigma + tau0_sq * I)
    Sigma_prop <- Sigma + tau0_sq * diag(d)
  
  # Second term: Pattern-specific likelihood terms
  for (j in 1:nrow(patterns)) {
    pattern<-patterns[j,]
    pattern_rows <- which(rowSums(unname(M==matrix(pattern, nrow = nrow(M), ncol = d, byrow = TRUE)))==d)
    nS<-length(pattern_rows)
    index<-1:d
    observedindex<-index[pattern==0]
    
    X_m<-X[pattern_rows,observedindex, drop=F]
    n_m<-nrow(X_m)
    
    key <- paste(pattern, collapse = "")
    
    # Pre-compute Cholesky of Sigma for efficient likelihood evaluation
    Sigma_chol[[key]] <- chol(Sigma[observedindex,observedindex])
    log_det_Sigma[[key]] <- 2 * sum(log(diag(Sigma_chol[[key]])))
    const_term[[key]] <- -0.5 * length(observedindex) * log(2 * pi)
    
    Sigma_prop_chol[[key]] <- chol(Sigma_prop[observedindex,observedindex])
    
    
  }
  

  

 
  
  # Main loop: iterate over observations (Line 1)
  for (i in 1:n) {
    
    index<-1:d
    observedindex<-index[!is.na(X[i,])]
    pattern<-M[i,]
    key <- paste(pattern, collapse = "")
    
    # Line 2: Remove observation i
    z[i] <- NA_integer_
    
    # Line 3: Count cluster sizes excluding i
    n_k <- tabulate(z[!is.na(z)], nbins = K)
    
    # Lines 4-11: Compute log-probabilities for existing clusters
    log_prob <- rep(-Inf, K + 1)
    
    for (k in 1:K) {
      if (n_k[k] > 0) {
        # Line 7: log n_k + log N(x_i | mu_k, Sigma)
        diff <- X[i, observedindex, drop=F] - mu_mat[k,observedindex, drop=F ]
        quad_form <- sum(backsolve(Sigma_chol[[key]], t(diff), transpose = TRUE)^2)
        log_prob[k] <- log(n_k[k]) + const_term[[key]] - 
          0.5 * (log_det_Sigma[[key]] + quad_form)
      }
      # else log_prob[k] stays -Inf (Line 9)
    }
    
    # Lines 12-14: New cluster probability
    # Line 13: Sample mu_new ~ N(mu0, Sigma + tau0_sq * I)
    mu_new <- drop(Rfast::rmvnorm(1, mu0, Sigma_prop))
    
    # Line 14: log alpha + log N(x_i^{(M_i)} | mu_new, Sigma)
    diff_new <- X[i, observedindex, drop=F ] - mu_new[observedindex]
    quad_form_new <- sum(backsolve(Sigma_chol[[key]], t(diff_new), transpose = TRUE)^2)
    log_prob[K + 1] <- log(alpha) + const_term[[key]] - 
      0.5 * (log_det_Sigma[[key]] + quad_form_new)
    
    # Line 15: p <- softmax(l_1, ..., l_{K+1})
    log_prob <- log_prob - max(log_prob)  # Numerical stability
    prob <- exp(log_prob)
    prob <- prob / sum(prob)
    
    # Line 16: Sample z_i ~ Categorical(p)
    z_new <- sample.int(K + 1, 1, prob = prob)
    
    # Lines 17-20: Create new cluster if needed
    if (z_new == K + 1) {
      # Line 18: K <- K + 1
      K <- K + 1
      # Line 19: mu_K <- mu_new
      mu_mat <- rbind(mu_mat, mu_new)
      z[i] <- K
    } else {
      z[i] <- z_new
    }
  }
  
  # Return updated state
  return(list(z = z, K = K, mu_mat = mu_mat))
}



UpdateCenters<-function(X, z,K, mu0, tau0_sq, Sigma, mu_mat){
  
  
  
   n_iter    = 200
   burnin    = 50
   
   mu_hat<-matrix(NA, nrow=K, ncol=ncol(X))
   
   for (k in 1:K){
     
  if (sum(z==k)>10){
  fit<-MHsamplerMu(X[z==k,, drop=F], Sigma, mu0=mu0, tau0_sq=tau0_sq, n_iter=n_iter, burnin=burnin, mu_init=mu_mat[k,])
  mu_hat[k,] <- fit$samples[n_iter,]
  }else{
    mu_hat[k,] <- mu_mat[k,] 
  }
   
   }
   
   return(mu_hat)
  
}

UpdateSharedSigma<-function(X,z,K,mu_mat, nu0, Psi0, Sigma_init){
  
  
   n_iter    = 200
   burnin    = 50
   fit<-MHsamplerSigma(X,z=z,mu=mu_mat, Psi0=Psi0, nu0=nu0, n_iter=n_iter, burnin=burnin,Sigma_init=Sigma_init)
   Sigma_hat <- fit$samples[, , n_iter]
   # post_samples <- fit$samples[, , (burnin + 1):n_iter]
  # Sigma_hat    <- apply(post_samples, c(1, 2), mean)
  
   return(Sigma_hat)
}

# ============================================================
# Algorithm 1: DP Gaussian Mixture with Shared Covariance
# (Only initialization + UpdateAssignmentsShared for now)
# ============================================================

dp_gmm_shared_Sigma <- function(X,
                                niter = 50,
                                nburn = 10,
                                alpha = 1.0,
                                mu0 = NULL,
                                tau0_sq = NULL,   # Prior variance for centers
                                nu0 = NULL,
                                Psi0 = NULL,
                                thin = 1,
                                verbose = TRUE) {
  
  ## Note the naive way we reference the different clusters is quite dangerous!
  
  n <- nrow(X)
  d <- ncol(X)
  
  # Set defaults
  if (is.null(mu0)| is.null(Psi0)){
  result <- optimize_mu_Sigma_constrained(X.NA)
  
  mu0<-result$mu_hat
  Psi0<-result$Sigma_hat
  }
  # if (is.null(mu0)){
  #   mu0 <- colMeans(X, na.rm = TRUE)
  #   }
  # 
  # if (is.null(Psi0)) {
  #   X_centered <- scale(X, center = TRUE, scale = FALSE)
  #   Psi0 <- crossprod(X_centered) / n + diag(d) * 1e-3
  # }
  
  if (is.null(tau0_sq)) tau0_sq <- 2.0   # Scalar: prior variance for mu_k
  if (is.null(nu0))    nu0 <- d + 2

  
  # ----------------------------------------
  # Lines 2-4: Initialize
  # ----------------------------------------
  
  # Initialize cluster assignments randomly
  z <- sample(1:5, n, replace = TRUE)
  K <- max(z)  # number of active clusters
  
  # Line 3: Sample mu_k ~ N(mu0, tau0_sq * I) for k = 1, ..., K
  mu_mat <- matrix(0, K, d)
  for (k in 1:K) {
    mu_mat[k, ] <- drop(Rfast::rmvnorm(1, mu=mu0, tau0_sq * diag(d)))
  }
  
  # Line 4: Sample Sigma ~ W^{-1}(Psi0, nu0)  -- single shared Sigma
  Sigma <- MCMCpack::riwish(nu0, Psi0)
  
  # ----------------------------------------
  # Storage
  # ----------------------------------------
  
  n_save <- floor((niter - nburn) / thin)
  z_samples <- matrix(0, nrow = n_save, ncol = n)
  K_samples <- numeric(n_save)
  Sigma_samples <- array(0, dim = c(n_save, d, d))
  mu_samples <- vector("list", n_save)
  weights_samples <- vector("list", n_save)
  
  if (verbose) {
    cat("Running MCMC for", niter, "iterations...\n")
    pb <- txtProgressBar(min = 0, max = niter, style = 3)
  }
  
  save_idx <- 1
  
  # ----------------------------------------
  # Line 5: Main MCMC loop
  # ----------------------------------------
  
  for (iter in 1:niter) {
    
    # Line 6: UpdateAssignmentsShared (Algorithm 2)
    result <- UpdateAssignmentsShared(X, z, K, mu_mat, Sigma, alpha,
                                      mu0, tau0_sq)
    z      <- result$z
    K      <- result$K
    mu_mat <- result$mu_mat
    
    # Line 7: Remove empty clusters; relabel; update K
    n_k <- tabulate(z, nbins = K)
    active <- n_k > 0
    
    if (sum(active) < K) {
      z      <- cumsum(active)[z]
      mu_mat <- mu_mat[active, , drop = FALSE]
      K      <- sum(active)
    }
    
    mu_mat <-UpdateCenters(X, z,K, mu0, tau0_sq, Sigma, mu_mat)
    # (X, z,K, mu0, tau0_sq, Sigma, mu_mat)
    Sigma <- UpdateSharedSigma(X,z,K,mu_mat, nu0, Psi0, Sigma_init=Sigma)
    
    # ----------------------------------------
    # Lines 10-12: Store samples after burn-in
    # ----------------------------------------
    
    if (iter > nburn && (iter - nburn) %% thin == 0) {
      n_k_active <- tabulate(z, nbins = K)
      z_samples[save_idx, ]    <- z
      K_samples[save_idx]      <- K
      Sigma_samples[save_idx, , ] <- Sigma
      mu_samples[[save_idx]]   <- lapply(1:K, function(k) mu_mat[k, ])
      weights_samples[[save_idx]] <- n_k_active / n
      save_idx <- save_idx + 1
    }
    
    if (verbose) setTxtProgressBar(pb, iter)
  }
  
  if (verbose) {
    close(pb)
    cat("\nMCMC completed!\n")
    cat("Average number of clusters:", mean(K_samples), "\n")
  }
  
  # Build density_samples for compatibility
  density_samples <- lapply(1:n_save, function(i) {
    list(
      mu      = mu_samples[[i]],
      Sigma   = Sigma_samples[i, , ],
      weights = weights_samples[[i]]
    )
  })
  
  return(list(
    z_samples       = z_samples,
    K_samples       = K_samples,
    Sigma_samples   = Sigma_samples,
    density_samples = density_samples,
    X               = X,
    model_type      = "shared_Sigma_independent_prior"
  ))
}


sample_posterior_predictive <- function(fit, n_pred = 1000) {
  
  n_samples <- length(fit$density_samples)
  X_pred <- matrix(0, nrow = n_pred, ncol = ncol(fit$X))
  
  for (i in 1:n_pred) {
    # Step 1: Randomly choose an MCMC iteration
    iter_idx <- sample(1:n_samples, 1)
    sample <- fit$density_samples[[iter_idx]]
    
    # Step 2: Sample from the mixture at this iteration
    K <- length(sample$mu)
    
    # 2a: Choose a component according to weights
    k <- sample(1:K, size = 1, prob = sample$weights)
    
    # 2b: Sample from that component
    X_pred[i, ] <- Rfast::rmvnorm(1,  sample$mu[[k]], sigma = sample$Sigma)
  }
  
  return(X_pred)
}

