functions
{
vector diagSPD_EQ( real rho, real L, int M) {
  return sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
}
matrix PHI(int N, int M, real L, vector x) {
  return sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
}
}
data
{
int<lower=1> N;      // number of observations
vector[N] x;         // univariate covariate
array[N] int y;         // target variable
        
real<lower=0> c_f;   // factor c to determine the boundary value L
int<lower=1> M_f;    // number of basis functions

real alpha_prior;
real beta_prior;
}
transformed data
{
real xmean = mean(x);
vector[N] xn = (x - xmean);
// Basis functions for f
real L_f = c_f*max(xn);
matrix[N,M_f] PHI_f = PHI(N, M_f, L_f, xn);
}

parameters{
vector[M_f] beta_f;          // the basis functions coefficients
real<lower=0> lengthscale_f; // lengthscale of f
real<lower = 0> alpha;

}
model{

lengthscale_f ~ inv_gamma(alpha_prior, beta_prior);
beta_f ~ std_normal();
alpha ~ normal(0,1);

// spectral densities
vector[M_f] diagSPD_f = diagSPD_EQ( lengthscale_f, L_f, M_f);

// Likelihood
y ~ neg_binomial_2(exp(PHI_f * (diagSPD_f .* beta_f)), alpha);
}