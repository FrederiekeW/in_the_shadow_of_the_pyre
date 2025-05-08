
// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> R;              // Number of regions
  int<lower=1> T;              // Number of time periods (decades)
  array[R, T] int<lower=0> trial_count; // Count of trials matrix (region x time)
  array[R] int<lower=1> N_neighbors;  // Number of neighbors per region
  array[R, max(N_neighbors)] int<lower=1, upper=R> neighbor_ids; // Neighbor IDs
}

parameters {
  real alpha;  // Overall intercept
  real beta;   // Effect of neighbor trials on region trials
  real<lower=0> sigma; // Std dev of region-specific intercepts
  array[R] real region_intercepts; // Region-specific intercepts
}

transformed parameters {
  array[R, T] real<lower=0> lambda; // Poisson rate (region x time)

  for (r in 1:R) {
    lambda[r, 1] = exp(alpha + region_intercepts[r]); // Initialize first time point
    for (t in 2:T) {
      real neighbor_influence = 0;
      for (n in 1:N_neighbors[r]) {
        int neighbor = neighbor_ids[r, n];
        neighbor_influence += trial_count[neighbor, t - 1];
      }
      lambda[r, t] = exp(alpha + beta * neighbor_influence + region_intercepts[r]);
    }
  }
}

model {
  // Priors
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ cauchy(0, 2);
  region_intercepts ~ normal(0, sigma);

  // Likelihood
  for (r in 1:R) {
    for (t in 2:T) {
      trial_count[r, t] ~ poisson(lambda[r, t]);
    }
  }
}
