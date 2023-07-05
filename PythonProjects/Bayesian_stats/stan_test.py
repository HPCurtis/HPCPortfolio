#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import stan as ps
import time



# data from Gossner et al (2022)
data = pd.read_csv("https://raw.githubusercontent.com/SimonErnesto/risk_dengue_2015-2019/main/dengue_travel_europe.csv")

infected = data["infected travellers"].values.astype("int") 
total = data["total travellers"].values.astype("int")
N = len(data)
country_id = data['codes'] = np.arange(1,100, dtype = int)

mod = '''
data{
int<lower=0> N;
int<lower = 1> N_countries;
array[N] int y_i;
array[N] int y_t;
array[N] int<lower = 1, upper = N_countries> country_id;
}
parameters{
vector<lower = 0>[4] hp; // hyperpriors
vector<lower = 0>[N_countries] T; 
vector<lower = 0>[N_countries] I; 
}
model{
hp ~ normal(0,1);
T ~ lognormal(hp[1], hp[2]);
I ~ lognormal(hp[3], hp[4]);
y_t ~ poisson(T[country_id]);
y_i ~ poisson(I[country_id]);
}
'''

datDic = {'N': N,
	  'y_i': infected,
	  'y_t': total,
	  'N_countries': N,
	  'country_id': country_id}


sm = ps.build(mod, data = datDic)
start_time = time.time()
fit = sm.sample(num_chains = 4, num_samples = 1000, num_warmup = 1000)
print("--- %s seconds ---" % (time.time() - start_time))


print(az.summary(fit))
#az.plot_pair(fit, var_names = ['I', 'T'])
#fig = plt.gcf() # to get the current figure...
#fig.savefig("pair.png") 

