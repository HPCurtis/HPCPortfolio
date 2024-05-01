import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, write_stan_json
import arviz as az
import matplotlib.pyplot as plt
import preliz as pz

def get_ig_params(x_vals, l_b=None, u_b=None, mass=0.95, plot=False):
    """
    Returns a weakly informative prior for the length-scale parameter of the GP kernel.
    """

    differences = np.abs(np.subtract.outer(x_vals, x_vals))
    if l_b is None:
        l_b = np.min(differences[differences != 0]) * 2
    if u_b is None:
        u_b = np.max(differences) / 1.5

    dist = pz.InverseGamma()
    pz.maxent(dist, l_b, u_b, mass, plot=plot)

    return dict(zip(dist.param_names, dist.params))

url = "https://raw.githubusercontent.com/aloctavodia/BAP3/main/code/data/islands.csv"
df = pd.read_csv(url)
df.head().round(1)
X = df[["lat", "lon"]].values

def haversine_distance(X, r=6.371):

    lat = np.radians(X[:,0])
    lon = np.radians(X[:,1])
    
    latd = lat[:,None] - lat
    lond = lon[:,None] - lon

    d = np.cos(lat[:,None]) * np.cos(lat)
    a = np.sin(latd / 2)** 2 + d * np.sin(lond / 2)** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return r * c

print(haversine_distance(X))

ivg_prior = get_ig_params(X)
model = CmdStanModel(stan_file="havershinegp.stan", cpp_options={'STAN_THREADS': 'TRUE'},
                     force_compile = True)

data = {'N': len(df),
        'k': X.shape[1],
        'x': X,
        'log_pop': df["logpop"].values,
        'y': df['total_tools'].values,
        'alpha_prior': ivg_prior["alpha"],
        'beta_prior': ivg_prior["beta"]}
write_stan_json("/home/harrison/Desktop/gitHubRepos/HPCStatsPortfolio/PythonProjects/Bayesian_stats/gaussianprocesses/havershinedata.json", data = data)                     
fit = model.sample("havershinedata.json", chains = 4 , iter_sampling=1000, parallel_chains = 4)