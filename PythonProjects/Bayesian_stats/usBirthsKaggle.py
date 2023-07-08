#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import stan as ps

# data from Gossner et al (2022)
data = pd.read_csv("https://raw.githubusercontent.com/HPCurtis/HPCStatsPortfolio/main/Data/us_births_2016_2021.csv")
# d2021 = df.iloc(df[, ])

# mod = '''
# data{
# int<lower=0> N;
# int<lower = 1> N_states;
# array[N] int y_i;
# array[N] int<lower = 1, upper = N_countries> state_id;
# }
# parameters{
# real hp; // 
# vector<lower = 0>[N_countries] T; 
# vector<lower = 0>[N_countries] I; 
# }
# model{
# hp ~ normal(0,1);
# T ~ lognormal(0, hp);
# y_t ~ poisson(T[country_id]);
# y_i ~ poisson(I[country_id]);
# }
# '''

# datDic = {'N': N,
# 	  'y_i': infected,
# 	  'y_t': total,
# 	  'N_countries': N,
# 	  'state_id': state_id}


# sm = ps.build(mod, data = datDic)
# fit = sm.sample(num_chains = 4, num_samples = 1000, num_warmup = 1000)



# print(az.summary(fit))


