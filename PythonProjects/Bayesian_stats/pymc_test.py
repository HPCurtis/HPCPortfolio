#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as at
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pymc.sampling_jax

os.chdir(os.getcwd())


# data from Gossner et al (2022)
data = pd.read_csv("https://raw.githubusercontent.com/SimonErnesto/risk_dengue_2015-2019/main/dengue_travel_europe.csv")

infected = data["infected travellers"].values.astype("int") 
total = data["total travellers"].values.astype("int")
N = len(data)

with pm.Model() as mod:
    mu_t = pm.HalfNormal("mu_t", 1)
    sd_t = pm.HalfNormal("sd_t", 1)
    mu_i = pm.HalfNormal("mu_i", 1)
    sd_i = pm.HalfNormal("sd_i", 1)
    I = pm.LogNormal("I", mu_i, sd_i, shape=N) #infected travellers
    T = pm.LogNormal("T", mu_t, sd_t, shape=N) #total travellers 
    y_i = pm.Poisson("y_i", mu=I, observed=infected)
    y_t = pm.Poisson("y_t", mu=T, observed=total)
    R = pm.Deterministic("R", 100000*I/T) #travellers rate of infection per 100000 travellers
    Ri = pm.Deterministic("Ri", (I/(T-I))*T) #raw risk (see Lee et al, 2021), i.e. ratio of infected and healthy travellers by total travellers 
    Rn = pm.Deterministic("Rn", Ri/Ri.max()) #normalised risk
    Rp = pm.Deterministic("Rp", Ri/Ri.sum()) #proportion of risk per country


with mod:
    idata = pymc.sampling_jax.sample_numpyro_nuts()

print(az.summary(idata))
