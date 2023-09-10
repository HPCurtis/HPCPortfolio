# #!/usr/bin/env python3

# The following is an conversion of the Bayesain Gaussian process analysis 
# script by Simon Erenesto that can be found at -
# https://github.com/SimonErnesto/Mental_health_spatiotemporal/blob/main/gp_model/gp_model.py
# orignally implemented using pymc but impleneted here using the stan PPL

# Import relevant analysis to run updated script
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
#import cmdstanpy as ps
import stan as ps

# Read in data csv and geespatial data in JSON format from github 
data = pd.read_csv("https://raw.githubusercontent.com/SimonErnesto/Mental_health_spatiotemporal/main/gp_model/mental_health_covid_data.csv")
geod = gpd.read_file('https://raw.githubusercontent.com/SimonErnesto/Mental_health_spatiotemporal/main/gp_model/south.geojson')

# Extract specific data coloumns and rename them.
data = data[['country__text__text', 'Date', 'GPS_Tot', 'Subregion']]
data.columns = ['country', 'time', 'score', 'region']

# remove all missing data from dataframe.
data['country'].replace(' ', np.nan, inplace=True)
data['region'].replace(' ', np.nan, inplace=True)
data = data.dropna()
data.reset_index(inplace=True)

date = []
nums = np.arange(10).astype('str')
for i in range(len(data.time)):
    d = data.time[i]
    if d[0] != '0' and d[1] not in nums:
        d2 = '0'+d[0]+d[1:]
        date.append(d2)
    else:
        date.append(d)
    
date2 = []
nums = np.arange(10).astype('str')
for i in range(len(date)):
    d = date[i]
    if d[-7] == '/':
        d2 = d[:-6]+'0'+d[-6:]
        date2.append(d2)
    else:
        date2.append(d)

data['date'] = date2

data = data[data.region=='213']
data.reset_index(inplace=True)

data = data.groupby(['date', 'country'], as_index=False).sum()

# Sort dataframe values by date.
data = data.sort_values('date')

D = len(data.date.unique())
C = len(data.country.unique())

# Convert country and date varaibles to integer data types
c = pd.factorize(data['country'])[0].astype('int32')
d = pd.factorize(data['date'])[0].astype('int32')

# Calculate # of unique values for d (dates) variable
ts = pd.unique(d)

score = data.score.values
# zscores = (score-score.mean())/score.std()

X = np.arange(len(d))[:,None]
X = np.ndarray.flatten(X)

# # Stan GP model
gpmod = '''
data{
int<lower=1> N;
array[N] real x;
array[N] int<lower=0> y;
}
transformed dat{
vector[N] jitter = rep_vector(1e-9, N);
}
parameters{
  real<lower=0> l;
  real<lower=0> sigma;
  vector[N] z_f;
}
model{

// Generate gp Kernel
matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));
                     
// Priors
l ~ normal(0, 1);
z_f ~ std_normal();
sigma ~ normal(0, 1);

//
f = l_f * z_f

// Likelihood
y ~ poisson(f);
}

'''
# dataDic = {'N': len(score[0:100]), 'x': X[0:100], 'y': score[0:100]}

# sm = ps.build(gpmod, data = dataDic)
# fit = sm.sample(num_chains = 1, num_samples = 100, num_warmup = 100)