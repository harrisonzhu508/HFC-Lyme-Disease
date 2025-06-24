#!/usr/bin/env python
"""
Kronecker-structured GP-NB via CmdStan.

Expects utils.load_train_test_data(seed, variables, count=True) to return
    X_train, y_train, X_val, y_val, X_test,
    df_test_meta, N_train, N_val, N_test, D,
    df_train_meta_train, df_val_meta
where each *meta* DF has ‘areanm’ and ‘year’.

Creates a full (area × year) grid, fills missing rows, and feeds the data
to stan_models/gp_kron.stan.
"""
# ───────────────────────────────────────────────────────────────────────────
import argparse, os, json
import numpy as np, pandas as pd
from cmdstanpy import CmdStanModel
from sklearn.metrics import mean_squared_error
from utils import (
    read_yaml,
    load_train_test_data,
    plot_predictions_with_ci,
    plot_coefficients,
)

# ───────────── CLI ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Kronecker GP-NB (CmdStan HMC)")
parser.add_argument("--seed", type=int, default=0)
args   = parser.parse_args()
seed   = args.seed

out_dir = f"results/GPR_hmc_kron_{seed}"
os.makedirs(out_dir, exist_ok=True)

# ───────────── 1. Variables & data splits ─────────────────────────────────
variables = read_yaml(os.path.join("config", "variables.yaml"))

(X_train, y_train,
 X_val,   y_val,
 X_test,  df_test_meta,
 N_train, N_val, N_test, D,
 df_train_meta_train, df_val_meta) = load_train_test_data(seed, variables, count=True)

def to_df(X, meta_df):
    dfx = pd.DataFrame(X, columns=variables).drop(columns=["year"])  # drop dup
    return pd.concat([meta_df.reset_index(drop=True)[["areanm", "year"]], dfx], axis=1)

df_tr  = to_df(X_train, df_train_meta_train);  df_tr["y"] = y_train
df_val = to_df(X_val,   df_val_meta);          df_val["y"] = y_val
df_te  = to_df(X_test,  df_test_meta);         df_te["y"] = np.nan  # no labels

# save these dfs to csv 
# df_tr.to_csv("train.csv")
# df_val.to_csv("val.csv")
# df_te.to_csv("test.csv")

# ───────────── 2. Build the full grid ─────────────────────────────────────
df_feat = (pd.concat([df_tr, df_val, df_te], ignore_index=True)
             .drop_duplicates(subset=["areanm", "year"], keep="first"))

areas = np.sort(df_feat["areanm"].unique())
years = np.sort(df_feat["year"].unique())

grid = (pd.MultiIndex.from_product([areas, years],
                                   names=["areanm", "year"])
          .to_frame(index=False))

df_grid = grid.merge(df_feat, on=["areanm", "year"], how="left", sort=True)

# ── patch any missing lat/lon ▸ look up the first non-null per council ────
latlon = (df_grid[["areanm", "latitude", "longitude"]]
            .dropna(subset=["latitude", "longitude"])
            .groupby("areanm")[["latitude", "longitude"]]
            .first())
df_grid[["latitude", "longitude"]] = (
    df_grid.set_index("areanm")[["latitude", "longitude"]]
           .combine_first(latlon)         # fills NaNs
           .reset_index(drop=True)
)
if df_grid[["latitude", "longitude"]].isna().any().any():
    raise ValueError("Some councils still have NaN lat/lon – please fix source data")

# flags & sentinel
df_grid["is_missing"] = df_grid["y"].isna().astype(int)
df_grid["y"] = df_grid["y"].fillna(0).astype(int)

# ───────────── 3. Design matrices in Kron order ───────────────────────────
N_space, N_year = len(areas), len(years)
N = N_space * N_year

X_space_levels = (df_grid[["areanm", "latitude", "longitude"]]
                    .drop_duplicates("areanm")
                    .sort_values("areanm")[["latitude", "longitude"]]
                    .to_numpy())                         # (N_space, 2)
year_levels    = years.astype(float)                    # (N_year,)

X_lin = df_grid[variables[3:]].to_numpy()               # (N, D_lin)
y_vec = df_grid["y"].to_numpy()
mask  = df_grid["is_missing"].to_numpy()

stan_data = {
    "N": N, "N_space": N_space, "N_year": N_year,
    "D_lin": X_lin.shape[1],
    "y": y_vec,
    "is_missing": mask,
    "X_space_levels": X_space_levels,
    "year_levels": year_levels,
    "X_lin": X_lin,
}

# ───────────── 4. Compile & sample ────────────────────────────────────────
model = CmdStanModel(stan_file="stan_models/gp_kron.stan")
fit   = model.sample(
            data=stan_data, seed=seed,
            chains=4, parallel_chains=4,
            iter_warmup=500, iter_sampling=2000, show_console=True)

post = fit.draws_pd()

# ───────────── 5. Posterior predictive: train rows ────────────────────────
train_cols = [c for c in post.columns if c.startswith("y_hat_train")]
y_hat_tr   = np.nanmean(post[train_cols].values, axis=0)
ci_lo_tr   = np.nanpercentile(post[train_cols].values,  2.5, axis=0)
ci_hi_tr   = np.nanpercentile(post[train_cols].values, 97.5, axis=0)

good_tr = (mask == 0)           # observed rows
plot_predictions_with_ci(
    y_true=y_vec[good_tr], y_pred_mean=y_hat_tr[good_tr],
    y_pred_lower=ci_lo_tr[good_tr], y_pred_upper=ci_hi_tr[good_tr],
    filepath=os.path.join(out_dir, "stan_gp_predictions_train.pdf"),
)

# ───────────── 6. Posterior predictive: validation split ──────────────────
val_cols = [c for c in post.columns if c.startswith("y_hat_val")]
if val_cols:
    y_hat_val = np.nanmean(post[val_cols].values, axis=0)
    mask_val  = ~np.isnan(y_hat_val)
    rmse_val  = np.sqrt(mean_squared_error(y_val[mask_val], y_hat_val[mask_val]))
else:
    print("Warning: y_hat_val not in Stan output → RMSE skipped")
    rmse_val  = float("nan")

# ───────────── 7. Coefficients (sorted high → low) ────────────────────────
lin_vars   = variables[3:]
coef_means = {v: float(post[f"beta[{i+1}]"].mean())
              for i, v in enumerate(lin_vars)}
sorted_coefs = dict(sorted(coef_means.items(), key=lambda kv: kv[1], reverse=True))

with open(os.path.join(out_dir, "results_gp_stan.json"), "w") as fp:
    json.dump({"validation_rmse": float(rmse_val),
               "coef_mean": sorted_coefs},
              fp, indent=2)

coef_df = (pd.DataFrame([coef_means])
             .loc[:, sorted_coefs.keys()])
plot_coefficients(coef_df,
                  title="Posterior mean β",
                  filepath=os.path.join(out_dir, "stan_gp_betas.pdf"))
