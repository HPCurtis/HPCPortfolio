import pymc as pm
#import aesara.tensor as pt
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import arviz as az
import geopandas as gpd
import matplotlib.pyplot as plt
import pymc.sampling_jax



data = pd.read_csv("https://raw.githubusercontent.com/SimonErnesto/Mental_health_spatiotemporal/main/gp_model/mental_health_covid_data.csv")
geod = gpd.read_file('https://raw.githubusercontent.com/SimonErnesto/Mental_health_spatiotemporal/main/gp_model/south.geojson')

data = data[['country__text__text', 'Date', 'GPS_Tot', 'Subregion']]

data.columns = ['country', 'date', 'score', 'region']

data['country'].replace(' ', np.nan, inplace=True)
data['region'].replace(' ', np.nan, inplace=True)
data = data.dropna()

data.reset_index(inplace=True)

date = []
nums = np.arange(10).astype('str')
for i in range(len(data.date)):
    d = data.date[i]
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
# data = data.groupby(['date'], as_index=False).agg({'country':'first', 'score':'mean'})

data = data.sort_values('date')


D = len(data.date.unique())
C = len(data.country.unique())
print(C)

c = pd.factorize(data['country'])[0].astype('int32')
d = pd.factorize(data['date'])[0].astype('int32')

ts = np.sqrt(pd.unique(d))


score = data.score.values
zscores = (score-score.mean())/score.std()

######### GRW LKJ Moedl ##############
with pm.Model() as mod:
    sd = pm.HalfNormal.dist(1.0)
    L, corr, std = pm.LKJCholeskyCov("L", n=C, eta=2.0, sd_dist=sd, compute_corr=True) 
    Σ = pm.Deterministic("Σ", L.dot(L.T))  
    w = pm.Normal('w', 0, 1.0, shape=(D,C))  
    σ = pm.HalfNormal('σ', 1.0)    
    β = pm.Deterministic('β', w.T*σ*ts)   
    B = pm.Deterministic('B', pm.math.matrix_dot(Σ,β))  
    α = pm.Normal('α', 0, 1.0, shape=C)  
    μ = pm.Deterministic('μ', α[c] + B[c,d]) 
    ϵ = pm.HalfNormal('ϵ', 1.0)
    y = pm.Normal("y", mu=μ, sigma=ϵ, observed=zscores)  

with mod:
    trace = pm.sample(10, tune=1, chains=1, cores=1, init='adapt_diag', target_accept=0.99)

print(az.summary(trace, var_names='Σ'))