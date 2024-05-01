
//  Normla mixture model as defined in chapter 6 of
// Osvaldo Martin BAP3 book. 
data {
  int<lower=1> K;          // number of mixture components
  int<lower=1> N;          // number of data points
  array[N] real y;         // observations
}
parameters {
  simplex[K] theta;          // mixing proportions
  ordered[K] mu;             // locations of mixture components
  real<lower=0> sigma;  // scales of mixture components
}
model {
  theta ~ dirichlet(rep_vector(1, K));
  vector[K] log_theta = log(theta);  // cache log calculation
  sigma ~ normal(0, 10);
  mu ~ normal(mean(y), 10);
  for (n in 1:N) {
    vector[K] lps = log_theta;
    for (k in 1:K) {
      lps[k] += normal_lpdf(y[n] | mu[k], sigma);
    }
    target += log_sum_exp(lps);
  }
}