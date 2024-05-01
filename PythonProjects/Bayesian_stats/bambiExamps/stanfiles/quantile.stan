data
{
int N;
int K;
matrix[N,K] X;
array[N] real y;
real tau;
vector[K] bs_sigma;
}
parameters
{
real intercept;
vector[K] beta;
real<lower=0> sigma;
}
transformed parameters
{
vector[N] mu;
mu = intercept + X * beta;
}
model
{
intercept ~ normal(0,4.7241);
for (k in 1:K){
    beta[k] ~ normal(0, bs_sigma[k]);
}
sigma ~ normal(0,1);
y ~ skew_double_exponential(mu, sigma, tau);
}