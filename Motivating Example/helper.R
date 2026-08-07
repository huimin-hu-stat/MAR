






library(mvtnorm)
library(MCMCpack)

# Batch log-density computation for multiple observations
dmvnorm_log_batch <- function(X, mu, Sigma_inv, log_det) {
  # X: n x d matrix
  # mu: d-vector
  # Sigma_inv: d x d precision matrix
  # log_det: scalar log-determinant
  
  d <- length(mu)
  n <- nrow(X)
  
  # Center data
  X_centered <- sweep(X, 2, mu, "-")
  
  # Compute quadratic forms for all observations at once
  # (X - mu)^T Sigma^{-1} (X - mu) for each row
  quad_forms <- rowSums((X_centered %*% Sigma_inv) * X_centered)
  
  # Return log-densities
  return(-0.5 * (d * log(2 * pi) + log_det + quad_forms))
}

# Batch posterior computation for cluster parameters
update_cluster_params_batch <- function(X, z, K, mu0, kappa0, nu0, Psi0_inv, d) {
  # Pre-allocate outputs
  mu_list <- vector("list", K)
  Sigma_list <- vector("list", K)
  Sigma_inv_list <- vector("list", K)
  log_det_list <- numeric(K)
  
  # Get cluster assignments as factor for faster operations
  z_factor <- factor(z, levels = 1:K)
  
  # Compute sufficient statistics for all clusters at once
  for (k in 1:K) {
    idx_k <- which(z == k)
    n_k <- length(idx_k)
    
    ###Continue here, do everything by pattern!!! ####
    
    if (n_k < 10) {
      # Empty cluster (shouldn't happen but be safe)
      mu_list[[k]] <- mu0
      Sigma_list[[k]] <- solve(Psi0_inv)
      Sigma_inv_list[[k]] <- Psi0_inv
      log_det_list[k] <- -determinant(Psi0_inv, logarithm = TRUE)$modulus[1]
      next
    }
    
    # Her the goal is to draw new Sigma and mu from the posterior for each cluster!
    X_k <- X[idx_k, , drop = FALSE]
    
    samples <- rejection_sampling_posterior(
      data=X_k,
      n_samples = 1,
      mu_n0 = mu0,
      lambda_n0 = lambda0,
      Psi_n0 = Psi0,
      nu_n0=nu0,
      Mup = Mopt
    )
    Sigma_list[[k]]<-samples$Sigma_samples[1,,]
    mu_list[[k]]<-samples$mu_samples[1,]
    
    # Pre-compute inverse and log-det
    Sigma_inv_list[[k]] <- solve(Sigma_list[[k]])
    log_det_list[k] <- determinant(Sigma_list[[k]], logarithm = TRUE)$modulus[1]
  }
  
  return(list(
    mu = mu_list,
    Sigma = Sigma_list,
    Sigma_inv = Sigma_inv_list,
    log_det = log_det_list
  ))
}

# Main optimized DP Gaussian mixture sampler
dp_gaussian_mixture_fast <- function(X, 
                                     niter = 5000,
                                     nburn = 1000,
                                     alpha = 1.0,
                                     mu0 = NULL,
                                     kappa0 = 0.01,
                                     nu0 = NULL,
                                     Psi0 = NULL,
                                     thin = 1,
                                     block_size = 50,  # Update observations in blocks
                                     verbose = TRUE) {
  
  n <- nrow(X)
  d <- ncol(X)
  
  # Set default hyperparameters
  if (is.null(mu0)) mu0 <- colMeans(X, na.rm = TRUE)
  if (is.null(nu0)) nu0 <- d + 2
  if (is.null(Psi0)) {
    X_centered <- scale(X, center = TRUE, scale = FALSE)
    Psi0 <- (crossprod(X_centered)) / n  # Faster than t(X) %*% X
    if (any(diag(Psi0) < 1e-6)) Psi0 <- Psi0 + diag(d) * 1e-3
  }
  
  Psi0_inv <- solve(Psi0)
  
  # Initialize with k-means
  if (verbose) cat("Initializing with k-means...\n")
  K_init <- min(5, max(2, floor(n/20)))
  km <- kmeans(X, centers = K_init, nstart = 10)
  z <- km$cluster
  K <- max(z)
  
  # Initialize cluster parameters using batch update
  params <- update_cluster_params_batch(X, z, K, mu0, kappa0, nu0, Psi0_inv, d)
  mu_list <- params$mu
  Sigma_list <- params$Sigma
  Sigma_inv_list <- params$Sigma_inv
  log_det_list <- params$log_det
  
  # Pre-allocate storage
  n_save <- floor((niter - nburn) / thin)
  z_samples <- matrix(0, nrow = n_save, ncol = n)
  K_samples <- numeric(n_save)
  density_samples <- vector("list", n_save)
  
  # Pre-allocate working matrices
  max_K <- min(50, n)  # Maximum expected clusters
  log_prob_matrix <- matrix(-Inf, nrow = block_size, ncol = max_K + 1)
  
  if (verbose) {
    cat("Running MCMC for", niter, "iterations (saving", n_save, "samples)...\n")
    pb <- txtProgressBar(min = 0, max = niter, style = 3)
  }
  
  save_idx <- 1
  
  # Create blocks for updating
  n_blocks <- ceiling(n / block_size)
  blocks <- split(1:n, cut(1:n, breaks = n_blocks, labels = FALSE))
  
  for (iter in 1:niter) {
    
    # Shuffle block order for better mixing
    block_order <- sample(blocks)
    
    # Step 1: Update cluster assignments in blocks
    for (block_idx in seq_along(block_order)) {
      block <- block_order[[block_idx]]
      n_block <- length(block)
      
      # Temporarily remove block observations
      z_block <- z[block]
      z[block] <- NA
      
      # Count cluster sizes (excluding current block)
      n_k <- tabulate(z[!is.na(z)], nbins = K)
      
      # Ensure we have enough columns in log_prob_matrix
      if (K + 1 > ncol(log_prob_matrix)) {
        extra_cols <- K + 1 - ncol(log_prob_matrix)
        log_prob_matrix <- cbind(log_prob_matrix, 
                                 matrix(-Inf, nrow = block_size, ncol = extra_cols))
      }
      
      # Reset log probabilities for this block
      log_prob_matrix[1:n_block, 1:(K+1)] <- -Inf
      
      # Compute log probabilities for existing clusters (VECTORIZED)
      X_block <- X[block, , drop = FALSE]
      
      for (k in 1:K) {
        if (n_k[k] > 0) {
          # Batch compute log-densities for all observations in block
          log_lik <- dmvnorm_log_batch(X_block, mu_list[[k]], 
                                       Sigma_inv_list[[k]], log_det_list[k])
          log_prob_matrix[1:n_block, k] <- log(n_k[k]) + log_lik
        }
      }
      
      # For new cluster, we still need to sample (can't vectorize perfectly)
      # But we can reuse the same new cluster params for the whole block
      Sigma_new <- MCMCpack::riwish(nu0, Psi0)
      mu_new <- drop(rmvnorm(1, mu0, Sigma_new / kappa0))
      Sigma_new_inv <- solve(Sigma_new)
      log_det_new <- determinant(Sigma_new, logarithm = TRUE)$modulus[1]
      
      log_lik_new <- dmvnorm_log_batch(X_block, mu_new, Sigma_new_inv, log_det_new)
      log_prob_matrix[1:n_block, K + 1] <- log(alpha) + log_lik_new
      
      # Sample cluster assignments for entire block
      for (i in 1:n_block) {
        # Normalize and sample (numerically stable)
        max_lp <- max(log_prob_matrix[i, 1:(K+1)])
        log_probs <- log_prob_matrix[i, 1:(K+1)] - max_lp
        probs <- exp(log_probs)
        probs <- probs / sum(probs)
        
        z_new <- sample(1:(K+1), 1, prob = probs)
        
        if (z_new == K + 1) {
          # Only create new cluster once per block
          if (!any(z[block[1:(i-1)]] == K + 1, na.rm = TRUE)) {
            K <- K + 1
            mu_list[[K]] <- mu_new
            Sigma_list[[K]] <- Sigma_new
            Sigma_inv_list[[K]] <- Sigma_new_inv
            log_det_list[K] <- log_det_new
          }
          z[block[i]] <- K
        } else {
          z[block[i]] <- z_new
        }
      }
    }
    
    # Remove empty clusters (vectorized)
    n_k <- tabulate(z, nbins = K)
    active <- n_k > 0
    
    if (sum(active) < K) {
      # Relabel using vectorized operations
      old_to_new <- cumsum(active)
      z <- old_to_new[z]
      
      mu_list <- mu_list[active]
      Sigma_list <- Sigma_list[active]
      Sigma_inv_list <- Sigma_inv_list[active]
      log_det_list <- log_det_list[active]
      K <- sum(active)
    }
    
    # Step 2: Update all cluster parameters at once (batch update)
    params <- update_cluster_params_batch(X, z, K, mu0, kappa0, nu0, Psi0_inv, d)
    mu_list <- params$mu
    Sigma_list <- params$Sigma
    Sigma_inv_list <- params$Sigma_inv
    log_det_list <- params$log_det
    
    # Store samples (with thinning)
    if (iter > nburn && (iter - nburn) %% thin == 0) {
      z_samples[save_idx, ] <- z
      K_samples[save_idx] <- K
      
      # Store density
      density_samples[[save_idx]] <- list(
        mu = lapply(mu_list, function(x) x),
        Sigma = lapply(Sigma_list, function(x) x),
        weights = tabulate(z, nbins = K) / n
      )
      
      save_idx <- save_idx + 1
    }
    
    if (verbose && iter %% 50 == 0) setTxtProgressBar(pb, iter)
  }
  
  if (verbose) {
    close(pb)
    cat("\nMCMC completed!\n")
    cat("Average number of clusters:", mean(K_samples), "\n")
    cat("Cluster range: [", min(K_samples), ",", max(K_samples), "]\n")
  }
  
  return(list(
    z_samples = z_samples,
    K_samples = K_samples,
    density_samples = density_samples,
    X = X,
    hyperparameters = list(
      alpha = alpha,
      mu0 = mu0,
      kappa0 = kappa0,
      nu0 = nu0,
      Psi0 = Psi0
    )
  ))
}











# ============================================================
# Rejection Sampling from Complex Posterior
# Using NIW as Proposal Distribution
# ============================================================







# Main DP Gaussian mixture sampler
dp_gaussian_mixture <- function(X, 
                                niter = 5000,
                                nburn = 1000,
                                alpha = 1.0,  # DP concentration parameter
                                # Base measure parameters (Normal-Inverse-Wishart)
                                mu0 = NULL,    # prior mean for centers
                                lambda0 = 0.01, # prior precision for centers
                                nu0 = NULL,    # prior df for Sigma
                                Psi0 = NULL) { # prior scale matrix for Sigma
  
  n <- nrow(X)
  d <- ncol(X)
  
  # Set default hyperparameters if not provided
  if (is.null(mu0)) mu0 <- colMeans(X, na.rm=T) #na.rm
  if (is.null(nu0)) nu0 <- d + 2
  if (is.null(Psi0)) Psi0 <- diag(d) * var(as.vector(X),na.rm=T)
  
  # Initialize cluster assignments randomly
  z <- sample(1:5, n, replace = TRUE)
  K <- max(z)  # number of active clusters
  
  # Initialize cluster parameters
  mu_list <- vector("list", K)
  Sigma_list <- vector("list", K)
  
  for (k in 1:K) {
    mu_list[[k]] <- mu0 + rnorm(d, 0, 1)
    Sigma_list[[k]] <- Psi0
  }
  
  # Storage for MCMC samples
  z_samples <- matrix(0, nrow = niter - nburn, ncol = n)
  K_samples <- numeric(niter - nburn)
  density_samples <- vector("list", niter - nburn)
  
  cat("Running MCMC for", niter, "iterations...\n")
  pb <- txtProgressBar(min = 0, max = niter, style = 3)
  
  index<-1:d
  
  
  Mopt<-find_optimal_M(X)
  
  for (iter in 1:niter) {
    
   
    
    # Step 1: Update cluster assignments for each observation:
    # This is the sampling from the DP prior!
    for (i in 1:n) {
      observedindex<-index[!is.na(X[i,])]
      
      # Remove observation i from its cluster
      k_old <- z[i]
      z[i] <- NA
      
      # Count observations in each cluster (excluding i)
      n_k <- table(factor(z, levels = 1:K))
      
      # Compute probabilities for existing clusters
      log_prob <- numeric(K + 1)  # +1 for potential new cluster
      
      for (k in 1:K) {
        if (n_k[k] > 0) {
          # Existing cluster with other members
          # This is prior*likelihood=proportional to posterior 
          log_prob[k] <- log(n_k[k]) + dmvnorm_log(X[i,observedindex], mu_list[[k]][observedindex], Sigma_list[[k]][observedindex,observedindex]) #Need to do this per pattern
        } else {
          # Empty cluster - skip it
          log_prob[k] <- -Inf
        }
      }
      
      # Probability of creating new cluster
      # Sample parameters from base measure
      Sigma_new <- rinvwishart(df=nu0, scale_matrix = Psi0) #riwish(v=nu0, S=Psi0)
      mu_new <- rmvnorm(1, mu0, Sigma_new / lambda0)
     
     # This is prior*likelihood=proportional to posterior  
      log_prob[K + 1] <- log(alpha) + dmvnorm_log(X[i,observedindex], mu_new[observedindex], Sigma_new[observedindex,observedindex])
      #Need to do this per pattern
      #+loglikGauss(data[i,], Sigma=Sigma_new, mu=mu_new)  
      
      # Normalize probabilities (in log space)
      log_prob <- log_prob - max(log_prob)  # for numerical stability
      prob <- exp(log_prob)
      prob <- prob / sum(prob)
      
      # Sample new cluster assignment
      z_new <- sample(1:(K+1), 1, prob = prob)
      
      if (z_new == K + 1) {
        # Create new cluster
        K <- K + 1
        mu_list[[K]] <- mu_new
        Sigma_list[[K]] <- Sigma_new
        z[i] <- K
      } else {
        z[i] <- z_new
      }
    }
    
    # Remove empty clusters
    n_k <- table(factor(z, levels = 1:K))
    active_clusters <- which(n_k > 0)
    if (length(active_clusters) < K) {
      # Relabel clusters
      z_old <- z
      z <- match(z, active_clusters)
      mu_list <- mu_list[active_clusters]
      Sigma_list <- Sigma_list[active_clusters]
      K <- length(active_clusters)
    }
    
    # Step 2: Update cluster parameters
    for (k in 1:K) {
      idx_k <- which(z == k)
      n_k <- length(idx_k)
      
      if (n_k > 0) { ##Need to do this per pattern!
        
        # Her the goal is to draw new Sigma and mu from the posterior for each cluster!
        X_k <- X[idx_k, , drop = FALSE]
        
        samples <- rejection_sampling_posterior(
          data=X_k,
          n_samples = 1,
          mu_n0 = mu0,
          lambda_n0 = lambda0,
          Psi_n0 = Psi0,
          nu_n0=nu0,
          Mup = Mopt
        )
        Sigma_list[[k]]<-samples$Sigma_samples[1,,]
        mu_list[[k]]<-samples$mu_samples[1,]
        
        # Sigma_list[[k]]<-diag(d)
        # mu_list[[k]]<-rep(0,d)
        
        #X_k <- X[idx_k, , drop = FALSE]
        #xbar_k <- colMeans(X_k)
        #
        # # Update Sigma using conjugate inverse-Wishart update
        # S_k <- t(X_k - matrix(xbar_k, n_k, d, byrow = TRUE)) %*% 
        #   (X_k - matrix(xbar_k, n_k, d, byrow = TRUE))
        # 
        # kappa_n <- lambda0 + n_k
        # nu_n <- nu0 + n_k
        # mu_n <- (lambda0 * mu0 + n_k * xbar_k) / kappa_n
        # 
        # Psi_n <- Psi0 + S_k + 
        #   (lambda0 * n_k / kappa_n) * outer(xbar_k - mu0, xbar_k - mu0)
        # 
        # # Sample new Sigma
        # Sigma_list[[k]] <- riwish(S=Psi_n, v=nu_n)
        # 
        # # Sample new mu given Sigma
        # mu_list[[k]] <- rmvnorm(1, mu_n, Sigma_list[[k]] / kappa_n)
      }
    }
    
    # Store samples after burn-in
    if (iter > nburn) {
      idx <- iter - nburn
      z_samples[idx, ] <- z
      K_samples[idx] <- K
      density_samples[[idx]] <- list(
        mu = mu_list,
        Sigma = Sigma_list,
        weights = as.numeric(table(z) / n)
      )
    }
    
    setTxtProgressBar(pb, iter)
  }
  close(pb)
  
  cat("\nMCMC completed!\n")
  cat("Average number of clusters:", mean(K_samples), "\n")
  
  return(list(
    z_samples = z_samples,
    K_samples = K_samples,
    density_samples = density_samples,
    X = X
  ))
}

# Function to evaluate density at grid points
evaluate_density <- function(fit, grid_x, grid_y, nsamples = 100) {
  # Use last nsamples from posterior
  total_samples <- length(fit$density_samples)
  sample_idx <- seq(total_samples - nsamples + 1, total_samples)
  
  ngrid <- length(grid_x)
  density_mat <- matrix(0, ngrid, ngrid)
  
  for (idx in sample_idx) {
    sample <- fit$density_samples[[idx]]
    K <- length(sample$mu)
    
    for (i in 1:ngrid) {
      for (j in 1:ngrid) {
        point <- c(grid_x[i], grid_y[j])
        dens <- 0
        
        for (k in 1:K) {
          dens <- dens + sample$weights[k] * 
            dmvnorm(point, mean = sample$mu[[k]], sigma = sample$Sigma[[k]])
        }
        
        density_mat[i, j] <- density_mat[i, j] + dens
      }
    }
  }
  
  density_mat <- density_mat / nsamples
  return(density_mat)
}

# Function to plot results
plot_density_estimate <- function(fit, X_test = NULL, ngrid = 50) {
  # Create grid
  x_range <- range(fit$X[,1]) + c(-1, 1) * diff(range(fit$X[,1])) * 0.2
  y_range <- range(fit$X[,2]) + c(-1, 1) * diff(range(fit$X[,2])) * 0.2
  
  grid_x <- seq(x_range[1], x_range[2], length.out = ngrid)
  grid_y <- seq(y_range[1], y_range[2], length.out = ngrid)
  
  cat("Evaluating density on grid...\n")
  dens <- evaluate_density(fit, grid_x, grid_y)
  
  # Plot
  par(mfrow = c(1, 2))
  
  # Contour plot
  contour(grid_x, grid_y, dens, 
          main = "Estimated Density (Contour)",
          xlab = "X1", ylab = "X2",
          nlevels = 15)
  points(fit$X[,1], fit$X[,2], pch = 20, cex = 0.5, col = "blue")
  
  # Image plot
  image(grid_x, grid_y, dens,
        main = "Estimated Density (Heatmap)",
        xlab = "X1", ylab = "X2",
        col = heat.colors(20))
  points(fit$X[,1], fit$X[,2], pch = 20, cex = 0.5)
  
  # Plot number of clusters over iterations
  par(mfrow = c(1, 1))
  plot(fit$K_samples, type = "l",
       main = "Number of Clusters Over Iterations",
       xlab = "Iteration (after burn-in)", ylab = "K")
  abline(h = mean(fit$K_samples), col = "red", lty = 2)
  
  invisible(dens)
}




# Main rejection sampling function
rejection_sampling_posterior <- function(data,n_samples,
                                         mu_n0, lambda_n0, Psi_n0, nu_n0,
                                         Mup = 0.5,  # Upper bound on density ratio
                                         max_attempts = 100000) {
  
  d <- length(mu_n0)
  n<-nrow(data)
  
  # Storage for accepted samples
  mu_samples <- matrix(0, n_samples, d)
  Sigma_samples <- array(0, dim = c(n_samples, d, d))
  
  n_accepted <- 0
  n_attempts <- 0
  acceptance_rate_history <- numeric(0)
  
  #cat("Starting rejection sampling...\n")
  #pb <- txtProgressBar(min = 0, max = n_samples, style = 3)
  
  while (n_accepted < n_samples && n_attempts < max_attempts) {
    n_attempts <- n_attempts + 1
    
    
    
    ##The only thing that really makes sense here, is for the proposal to be the prior; We need to utilize the full likelihood.
    
    # Step 1: Sample from proposal, prior!! (NIW)
    Sigma_prop <- rinvwishart(df=nu_n0, scale_matrix = Psi_n0) #riwish(S=Psi_n0, v=nu_n0)
    mu_prop <- rmvnorm(1, mu_n0, Sigma_prop / lambda_n0)
    
    
    # # Step 2: Evaluate densities
    # target_density <- evaluate_target_posterior(data, mu_prop, Sigma_prop, 
    #                                             mu_n0, lambda_n0, Psi_n0, nu_n0)
    # 
    # #(data, mu, Sigma, 
    # #mu_n0, lambda_n0, Psi_n0, nu_n0)
    # proposal_density <- evaluate_niw_proposal(data, mu_prop, Sigma_prop, 
    #                                           mu_n0, lambda_n0, Psi_n0, nu_n0)
    
    # Step 3: Compute acceptance ratio
    #Since we use the prior as a proposal, the ratio is simply given as 1/M * Likelihood
    #ratio <- exp(target_density - (log(M) + proposal_density))
    ratio <- exp(loglikGauss(data, Sigma_prop, mu_prop)/n - log(Mup) )
  
    
    # if (proposal_density > 0) {
    #   
    # } else {
    #   ratio <- 0
    # }
    
    # Step 4: Accept/reject
    u <- runif(1)
    if (u < ratio) {
      n_accepted <- n_accepted + 1
      mu_samples[n_accepted, ] <- mu_prop
      Sigma_samples[n_accepted, , ] <- Sigma_prop
      
      #setTxtProgressBar(pb, n_accepted)
    }
    
    # Track acceptance rate every 1000 iterations
    if (n_attempts %% 1000 == 0) {
      acceptance_rate_history <- c(acceptance_rate_history, n_accepted / n_attempts)
    }
  }
  
  #close(pb)
  
  if (n_accepted < n_samples) {
    warning(paste("Only obtained", n_accepted, "samples out of", n_samples, "requested"))
    mu_samples <- mu_samples[1:n_accepted, ]
    Sigma_samples <- Sigma_samples[1:n_accepted, , ]
  }
  
  #cat("\nRejection sampling completed!\n")
  #cat("Acceptance rate:", n_accepted / n_attempts, "\n")
  #cat("Total attempts:", n_attempts, "\n")
  
  return(list(
    mu_samples = mu_samples,
    Sigma_samples = Sigma_samples,
    acceptance_rate = n_accepted / n_attempts,
    acceptance_rate_history = acceptance_rate_history,
    n_attempts = n_attempts
  ))
}


# Main rejection sampling function
rejection_sampling_posterior_mu <- function(data,n_samples,
                                         mu_n0, lambda_n0, Sigma_prop,
                                         Mup = 0.5,  # Upper bound on density ratio
                                         max_attempts = 100000) {
  
  # Update mu given Sigma!
  
  d <- length(mu_n0)
  
  # Storage for accepted samples
  mu_samples <- matrix(0, n_samples, d)
  
  n_accepted <- 0
  n_attempts <- 0
  acceptance_rate_history <- numeric(0)
  
  #cat("Starting rejection sampling...\n")
  #pb <- txtProgressBar(min = 0, max = n_samples, style = 3)
  
  while (n_accepted < n_samples && n_attempts < max_attempts) {
    n_attempts <- n_attempts + 1
    
    
    
    ##The only thing that really makes sense here, is for the proposal to be the prior; We need to utilize the full likelihood.
    
    # Step 1: Sample from proposal, prior!! (NIW)
    #Sigma_prop <- rinvwishart(df=nu_n0, scale_matrix = Psi_n0) #riwish(S=Psi_n0, v=nu_n0)
    mu_prop <- rmvnorm(1, mu_n0, Sigma_prop / lambda_n0)
    
  
    # Step 3: Compute acceptance ratio
    #Since we use the prior as a proposal, the ratio is simply given as 1/M * Likelihood
    #ratio <- exp(target_density - (log(M) + proposal_density))
    ratio <- exp(loglikGauss(data, Sigma_prop, mu_prop)/n - log(Mup) )
    
  
    
    # Step 4: Accept/reject
    u <- runif(1)
    if (u < ratio) {
      n_accepted <- n_accepted + 1
      mu_samples[n_accepted, ] <- mu_prop
      
      #setTxtProgressBar(pb, n_accepted)
    }
    
    # Track acceptance rate every 1000 iterations
    if (n_attempts %% 1000 == 0) {
      acceptance_rate_history <- c(acceptance_rate_history, n_accepted / n_attempts)
    }
  }
  
  #close(pb)
  
  if (n_accepted < n_samples) {
    warning(paste("Only obtained", n_accepted, "samples out of", n_samples, "requested"))
    mu_samples <- mu_samples[1:n_accepted, ]
  }
  
  #cat("\nRejection sampling completed!\n")
  #cat("Acceptance rate:", n_accepted / n_attempts, "\n")
  #cat("Total attempts:", n_attempts, "\n")
  
  return(list(
    mu_samples = mu_samples,
    acceptance_rate = n_accepted / n_attempts,
    acceptance_rate_history = acceptance_rate_history,
    n_attempts = n_attempts
  ))
}


############################Adapt this code ####################################################################

# ============================================================
# Rejection Sampling for Shared Sigma
# ============================================================



# Log-likelihood (product over clusters)
log_likelihood_Sigma <- function(Sigma, X, z, mu_mat) {
  n <- nrow(X)
  d <- ncol(X)
  K <- nrow(mu_mat)
  
  # Compute total scatter
  S_total <- matrix(0, d, d)
  
  for (k in 1:K) {
    idx_k <- which(z == k)
    n_k <- length(idx_k)
    
    if (n_k > 0) {
      X_k <- X[idx_k, , drop = FALSE]
      mu_k <- mu_mat[k, ]
      
      # Scatter for cluster k
      X_centered <- X_k - matrix(mu_k, n_k, d, byrow = TRUE)
      S_k <- crossprod(X_centered)
      S_total <- S_total + S_k
    }
  }
  
  # Log-likelihood
  log_det_Sigma <- determinant(Sigma, logarithm = TRUE)$modulus[1]
  
  log_lik <- -n/2 * log_det_Sigma
  log_lik <- log_lik - 0.5 * sum(diag(S_total %*% solve(Sigma)))
  
  return(log_lik)
}

# Log-posterior (target)
log_posterior_Sigma <- function(Sigma, X, z, mu_mat, Psi0, nu0) {
  log_prior <- log_prior_Sigma(Sigma, Psi0, nu0)
  log_lik <- log_likelihood_Sigma(Sigma, X, z, mu_mat)
  return(log_prior + log_lik)
}

# Rejection sampling
rejection_sample_Sigma <- function(X, z, mu_mat, Psi0, nu0, 
                                   M = 2.0, max_attempts = 1000) {
  
  d <- ncol(X)
  n <- nrow(X)
  
  # Proposal: Use the conjugate posterior (this is cheating, but illustrative)
  # In practice, you'd use a different proposal
  
  # Compute posterior parameters
  S_total <- matrix(0, d, d)
  for (k in 1:nrow(mu_mat)) {
    idx_k <- which(z == k)
    if (length(idx_k) > 0) {
      X_centered <- X[idx_k, , drop = FALSE] - 
        matrix(mu_mat[k, ], length(idx_k), d, byrow = TRUE)
      S_total <- S_total + crossprod(X_centered)
    }
  }
  
  nu_post <- nu0 + n
  Psi_post <- Psi0 + S_total
  
  # Proposal distribution
  proposal_sample <- function() {
    MCMCpack::riwish(nu_post, Psi_post)
  }
  
  proposal_density <- function(Sigma) {
    log_det <- determinant(Sigma, logarithm = TRUE)$modulus[1]
    log_dens <- -(nu_post + d + 1)/2 * log_det
    log_dens <- log_dens - 0.5 * sum(diag(Psi_post %*% solve(Sigma)))
    return(exp(log_dens))
  }
  
  # Rejection sampling loop
  for (attempt in 1:max_attempts) {
    # Sample from proposal
    Sigma_prop <- proposal_sample()
    
    # Compute acceptance ratio
    target <- exp(log_posterior_Sigma(Sigma_prop, X, z, mu_mat, Psi0, nu0))
    proposal <- proposal_density(Sigma_prop)
    
    ratio <- target / (M * proposal)
    
    # Accept/reject
    if (runif(1) < ratio) {
      return(list(Sigma = Sigma_prop, accepted = TRUE, attempts = attempt))
    }
  }
  
  # Failed to accept
  return(list(Sigma = NULL, accepted = FALSE, attempts = max_attempts))
}

# ============================================================
# Alternative: Metropolis-Hastings (More Practical)
# ============================================================

metropolis_hastings_Sigma <- function(Sigma_current, X, z, mu_mat, 
                                      Psi0, nu0, proposal_scale = 0.1) {
  
  d <- ncol(X)
  
  # Proposal: Random walk on Cholesky factor
  L_current <- chol(Sigma_current)
  
  # Perturb Cholesky (ensures positive definiteness)
  L_prop <- L_current + matrix(rnorm(d^2, 0, proposal_scale), d, d)
  L_prop[lower.tri(L_prop)] <- 0  # Keep upper triangular
  
  # Ensure positive diagonal
  diag(L_prop) <- abs(diag(L_prop)) + 1e-6
  
  Sigma_prop <- t(L_prop) %*% L_prop
  
  # Compute log acceptance ratio
  log_ratio <- log_posterior_Sigma(Sigma_prop, X, z, mu_mat, Psi0, nu0) -
    log_posterior_Sigma(Sigma_current, X, z, mu_mat, Psi0, nu0)
  
  # Jacobian correction for Cholesky parameterization
  # log|J| = sum(log(diag(L_prop))) - sum(log(diag(L_current)))
  log_jacobian <- sum(log(abs(diag(L_prop)))) - sum(log(abs(diag(L_current))))
  log_ratio <- log_ratio + log_jacobian
  
  # Accept/reject
  if (log(runif(1)) < log_ratio) {
    return(list(Sigma = Sigma_prop, accepted = TRUE))
  } else {
    return(list(Sigma = Sigma_current, accepted = FALSE))
  }
}


########################Adapt this code ########################################



























optimize_mu_Sigma_constrained <- function(data, mu_init = NULL, Sigma_init = NULL) {
  
  n <- nrow(data)
  p <- ncol(data)
  
  if (is.null(mu_init)) mu_init <- colMeans(data, na.rm=T)
  if (is.null(Sigma_init)) Sigma_init <- diag(p)
  
  # Parameterize Sigma via Cholesky: Sigma = L L^T
  # This ensures positive definiteness
  L_init <- chol(Sigma_init)
  L_vec <- L_init[upper.tri(L_init, diag = TRUE)]
  
  theta_init <- c(mu_init, L_vec)
  
  # Objective function
  objective <- function(theta) {
    mu <- theta[1:p]
    L_vec <- theta[(p+1):length(theta)]
    
    # Reconstruct L (upper triangular)
    L <- matrix(0, p, p)
    L[upper.tri(L, diag = TRUE)] <- L_vec
    
    # Compute Sigma
    Sigma <- t(L) %*% L
    
    -loglikGauss(data, Sigma, mu)/n
  }
  
  result <- optim(
    par = theta_init,
    fn = objective,
    method = "BFGS",
    control = list(maxit = 5000)
  )
  
  # Extract results
  mu_hat <- result$par[1:p]
  L_vec <- result$par[(p+1):length(result$par)]
  L <- matrix(0, p, p)
  L[upper.tri(L, diag = TRUE)] <- L_vec
  Sigma_hat <- t(L) %*% L
  
  list(
    mu_hat = mu_hat,
    Sigma_hat = Sigma_hat,
    loglik = -result$value,
    convergence = result$convergence
  )
}


