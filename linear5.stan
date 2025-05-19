data {
  int<lower=1> R;                          // Number of regions
  int<lower=1> T;                          // Number of time steps
  int<lower=1> max_neighbors;             // Maximum number of neighbors
  array[R, T] int<lower=0> trial_count;   // Trial counts per region and time
  array[R] int<lower=0> N_neighbors;      // Number of neighbors per region
  array[R, max_neighbors] int<lower=1> neighbor_ids; // Neighbor indices (padded)
}

parameters {
  real<lower=1e-3> sigma_gr;              // Between-region variability
  vector[R] gr_raw;                       // Standardized regional growth deviations
  real<lower=1e-3> sigma;                 // Observation noise std dev
  real beta;                              // Neighbor influence coefficient
}

transformed parameters {
  vector[R] gr;
  gr = sigma_gr * gr_raw;                // Region-specific growth rates (no overalll intercept)
}

model {
  // Priors
  gr_raw ~ normal(0, 1);                 
  sigma_gr ~ exponential(1);
  sigma ~ exponential(1);
  beta ~ normal(0, 1);

  // Likelihood
  for (r in 1:R) {
    for (t in 2:T) {
      real neighbor_sum = 0;
      for (n in 1:N_neighbors[r]) {
        neighbor_sum += log1p(trial_count[neighbor_ids[r, n], t - 1]);
      }

      real mu = log1p(trial_count[r, t - 1]) + gr[r] + beta * neighbor_sum;

      target += lognormal_lpdf(trial_count[r, t] + 1 | mu, sigma);
    }
  }
}
