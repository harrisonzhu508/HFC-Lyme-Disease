# ───────────────────────────────────────────────────────────────────────────
import argparse, os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use GPU 0
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
import sys 
sys.path.append("/home/hbz15/lyme_disease_working/submission/scripts/")
from utils import (
    read_yaml,
    load_train_test_data,
    plot_predictions_with_ci,
    plot_coefficients,
)

# ───────────── CLI ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run GP regression with hmc.")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
args = parser.parse_args()
seed = args.seed
np.random.seed(seed)
os.makedirs(f"results/GPR_{seed}", exist_ok=True)


out_dir = f"results/GPR_hmc_kron_pymc_{seed}"
os.makedirs(out_dir, exist_ok=True)

# ───────────── 1. Variables & data splits ─────────────────────────────────
variables = read_yaml(os.path.join("config", "variables.yaml"))

(X_train, y_train,
 X_val,   y_val,
 X_test,  df_test_meta,
 N_train, N_val, N_test, D,
 df_train_meta_train, df_val_meta, scaler) = load_train_test_data(seed, variables, count=True)

def to_df(X, meta_df):
    dfx = pd.DataFrame(X, columns=variables).drop(columns=["year"])  # drop dup
    return pd.concat([meta_df.reset_index(drop=True)[["areanm", "year"]], dfx], axis=1)

df_tr  = to_df(X_train, df_train_meta_train);  df_tr["y"] = y_train
df_val = to_df(X_val,   df_val_meta);          df_val["y"] = y_val
df_te  = to_df(X_test,  df_test_meta);         df_te["y"] = np.nan  # no labels

# merge df_tr, df_val, df_te
df = pd.concat([df_tr, df_val, df_te], ignore_index=True)
df["train"] = np.concatenate([np.ones(N_train), np.zeros(N_val + N_test)])
df["val"] = np.concatenate([np.zeros(N_train), np.ones(N_val), np.zeros(N_test)])
df["test"] = np.concatenate([np.zeros(N_train + N_val), np.ones(N_test)])

# then sort by area, year
df = df.sort_values(by=["areanm", "year"]).reset_index(drop=True)
# remove city of london
df = df[df["areanm"] != "City of London"].reset_index(drop=True)
all_areanms = df["areanm"].unique()
all_years = df["year"].unique()

# check for all years, all areanms are present
for year in all_years:
    this_df = df[df["year"] == year]
    print(f"Year {year}: {len(this_df['areanm'].unique())} areas, expected {len(all_areanms)}")
    # which area is missing?
    missing_areanms = set(all_areanms) - set(this_df["areanm"].unique())
    if missing_areanms:
        print(f"  Missing areas: {missing_areanms}")


# create df_space by removing year rows and keep longtitude, latitude
df_space = df[df["year"]==2017][["longitude", "latitude"]].copy()
X_space = df_space.values
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
years = np.array(years)

import numpy as np
import pandas as pd
import pymc.sampling.jax as pmjax
import pymc as pm
from pymc.gp.cov import Matern32, WhiteNoise

# ─── 1) Data prep ────────────────────────────────────────────────────────────
y_obs      = df["y"].to_numpy()
mask_train = df["train"].to_numpy(bool)
mask_valid = df["val"].to_numpy(bool)
mask_test  = (~mask_train) & (~mask_valid)
X_lin = df[variables[3:]].to_numpy()

N_space = X_space.shape[0]          # (N_space, 2)
N_year  = years.shape[0]            # (N_year, 1)
N       = N_space * N_year          # total grid size

# ─── 2) Build & fit the model ───────────────────────────────────────────────
with pm.Model() as model:
    # hyper-priors
    σ       = pm.HalfNormal("σ", sigma=1.0)
    ℓ_space = pm.HalfNormal("ℓ_space", sigma=1.0)
    ℓ_year  = pm.HalfNormal("ℓ_year",  sigma=1.0)
    σ_n     = pm.HalfNormal("σ_n",     sigma=0.1)

    # separate kernels ( *no* manual Kron object here)
    K_space = σ**2 * Matern32(input_dim=2, ls=ℓ_space)
    K_year  =        Matern32(input_dim=1, ls=ℓ_year)

    # Latent GP with Kronecker structure
    gp = pm.gp.LatentKron(cov_funcs=[K_space, K_year])

    # full latent field on the space-time grid
    f_all = gp.prior(
        "f_all",
        Xs=[
            X_space,                 # (N_space, 2)
            years.reshape(-1, 1)     # (N_year, 1)
        ]
    )

    # add linear effects
    f_all += pm.math.dot(X_lin, pm.Normal("β", mu=0, sigma=1, shape=X_lin.shape[1]))

    # training slice
    η_train = f_all[mask_train]
    y_train = y_obs[mask_train]

    φ = pm.Gamma("phi", alpha=2.0, beta=0.1)

    pm.NegativeBinomial(
        "y_obs",
        mu=pm.math.exp(η_train),
        alpha=φ,
        observed=y_train,
    )

    idata = pmjax.sample_numpyro_nuts(     # ←–––– just this one function
        draws=2_000,
        tune=1_000,
        chains=4,
        chain_method="vectorized",         # all chains in a single JIT
        nuts_sampler="numpyro",
        target_accept=0.90,
        progressbar=True,                 # avoids Python I/O inside JIT
    )

# ─── 3) Posterior predictive on held-out data ───────────────────────────────
with model:
    ppc = pm.sample_posterior_predictive(
        idata,                        # pass the *idata*, not the old trace
        var_names=["f_all", "phi"],
        keep_size=True,
    )

# reshape & slice
f_all_samps = ppc["f_all"]                # (draw, N)
phi_samps   = ppc["phi"][:, None]         # (draw, 1)  for broadcasting

f_val_samps  = f_all_samps[:, mask_valid]  # (draw, N_val)
f_test_samps = f_all_samps[:, mask_test]   # (draw, N_test)

# Negative-binomial simulations
mu_val  = np.exp(f_val_samps)
mu_test = np.exp(f_test_samps)

y_val_pred = np.random.negative_binomial(
    n=phi_samps,
    p=phi_samps / (phi_samps + mu_val),
)

y_test_pred = np.random.negative_binomial(
    n=phi_samps,
    p=phi_samps / (phi_samps + mu_test),
)

# save samples
np.savez(os.path.join(out_dir, "samples.npz"),
         f_all=f_all_samps,
         phi=phi_samps,
         mu_val=mu_val,
         mu_test=mu_test,
         y_val_pred=y_val_pred,
         y_test_pred=y_test_pred)


