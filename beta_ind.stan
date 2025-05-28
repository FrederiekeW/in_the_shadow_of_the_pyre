data {
  int<lower=1> R;                          // Number of regions
  int<lower=1> T;                          // Number of time steps
  int<lower=1> max_neighbors;             // Max number of neighbors
  array[R, T] int<lower=0> trial_count;   // Trial counts per region and time
  array[R] int<lower=0> N_neighbors;      // Number of neighbors per region
  array[R, max_neighbors] int<lower=1> neighbor_ids; // Neighbor indices
}

parameters {
  real<lower=1e-3> sigma_gr;              // Between-region variability
  vector[R] gr_raw;                       // regional growth deviations
  real<lower=1e-3> sigma;                 //  Variation
  matrix[R, R] beta_mat;                  // Influence: from region to region
}

transformed parameters {
  vector[R] gr;
  gr = sigma_gr * gr_raw;                // Final regional growth terms
}

model {
  // Priors
  gr_raw ~ normal(0, 1);
  sigma_gr ~ exponential(1);
  sigma ~ exponential(1);

  to_vector(beta_mat) ~ normal(0, 1); 

  // Likelihood
  for (r in 1:R) {
    for (t in 2:T) {
      real neighbor_sum = 0;
      for (n in 1:N_neighbors[r]) {
        int neighbor = neighbor_ids[r, n];
        neighbor_sum += beta_mat[neighbor, r] * log1p(trial_count[neighbor, t - 1]);
      }

      real mu = log1p(trial_count[r, t - 1]) + gr[r] + neighbor_sum;
      target += lognormal_lpdf(trial_count[r, t] + 1 | mu, sigma);
    }
  }
}
