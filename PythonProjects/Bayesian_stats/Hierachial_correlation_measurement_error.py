# Import relevant data analysis and visualisation packages.
import stan as ps
import numpy as np
import os
import pandas as pd
from patsy import dmatrix
import arviz as az
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

# Import data from notebook github exported from the RData files from original Win BUGS example
observed = 'https://raw.githubusercontent.com/ebrlab/Statistical-methods-for-research-workers-bayes-for-psychologists-and-neuroscientists/master/wip/Data/observed.csv'
epsilon = 'https://raw.githubusercontent.com/ebrlab/Statistical-methods-for-research-workers-bayes-for-psychologists-and-neuroscientists/master/wip/Data/epsilon.csv'

# Create separate panda dataframes for observed parameter estmates and associated error in epsilon
df1 = pd.read_csv(observed)
df2 = pd.read_csv(epsilon)

# Generate matricies for estimated parameter values and measuremnt error values.
y = np.asarray(dmatrix("0 + theta + beta", data = df1) )
epsilon = np.asarray(dmatrix("0 + theta_epsilon + beta_epsilon", data = df2) )

# Stan implentation of the Matske et al. model with LKJ priors in non centered parameterisation

hier = """
data {
// Stan version of "Bayesian Inference for Correlations in the Presence of Measurement Error 
// and Estimation Uncertainty" WinBugs model

int<lower = 1> n; // Number of observations
int<lower = 1> J; // Number of groups

// Wide format data array containing 
// theta and beta observed values. This allows for vectorisation
// for more efficent sampling
vector[J] y[n];
vector[J] epsilon[n];

// Prior values must be integer as that is how ther specified in python
// to use real number these would need changing
int sigma_mu_theta;
int sigma_mu_beta;
int sigma_sigma_theta;
int sigma_sigma_beta;
int cor_val;
}
parameters {
  vector[J] mu;
  vector<lower = 0>[J] sigma;   
  vector[J] z[n];
  cholesky_factor_corr[J] rho;
}

transformed parameters {
// Non-centered parameterisation
vector[J] eta[n];
matrix[J, J] L = diag_pre_multiply(sigma, rho);
for (i in 1:n){
  eta[i,] = mu + L * z[i,];
    }
}

model{

//Priors
// Hyperpriors
mu ~ normal(0, sigma_mu_theta);
sigma ~ normal(0, sigma_sigma_theta);
rho ~ lkj_corr_cholesky(cor_val);

for(i in 1:n) {
    z[i,] ~ std_normal();
}

// likelihood
for (i in 1:n){
y[i,] ~ normal(eta[i,], epsilon[i, ]);  
}

}

generated quantities {
// Reassemble correlation matrix after cholesky decomposition.
  corr_matrix[J] rho_u = rho * rho';
}
"""

data = {'n': len(y),
'J': 2,
'y': y,
'epsilon': epsilon,
'sigma_sigma_theta': 5,
'sigma_sigma_beta': 5,
'sigma_mu_theta': 2,
'sigma_mu_beta': 2,
'cor_val': 1}

sm1 = ps.build(hier, data=data, random_seed=1)

fit = sm1.hmc_nuts_diag_e_adapt(num_chains=4, stepsize = 1)
fit_df = fit.to_frame()
sum = az.summary(fit)
print(sum)