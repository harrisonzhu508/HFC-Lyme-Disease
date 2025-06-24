import argparse
parser = argparse.ArgumentParser(description="Run GPR regression HMC")
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
args = parser.parse_args()
seed = args.seed
base_dir = f"results/GPR_hmc_{seed}"


import os 
import json
os.makedirs(f"results/GPR_hmc_{seed}", exist_ok=True)
import numpy as np
from sklearn.metrics import mean_squared_error
from cmdstanpy import CmdStanModel
from utils import (
    read_yaml,
    load_train_test_data,
    plot_coefficients,
    plot_predictions_with_ci,
)
import pandas as pd

# 1. load your variable list (or define in code)
variables = [
    'year',
    'latitude',
    'longitude',
    't2m_1',
    't2m_2',
    't2m_3',
    't2m_4',
    't2m_5',
    't2m_6',
    't2m_7',
    't2m_8',
    't2m_9',
    't2m_10',
    't2m_11',
    't2m_12',
    'spec_humidity_1',
    'spec_humidity_2',
    'spec_humidity_3',
    'spec_humidity_4',
    'spec_humidity_5',
    'spec_humidity_6',
    'spec_humidity_7',
    'spec_humidity_8',
    'spec_humidity_9',
    'spec_humidity_10',
    'spec_humidity_11',
    'spec_humidity_12',
    'rel_humidity_1',
    'rel_humidity_2',
    'rel_humidity_3',
    'rel_humidity_4',
    'rel_humidity_5',
    'rel_humidity_6',
    'rel_humidity_7',
    'rel_humidity_8',
    'rel_humidity_9',
    'rel_humidity_10',
    'rel_humidity_11',
    'rel_humidity_12',
    'src_1',
    'src_2',
    'src_3',
    'src_4',
    'src_5',
    'src_6',
    'src_7',
    'src_8',
    'src_9',
    'src_10',
    'src_11',
    'src_12',
    'sp_1',
    'sp_2',
    'sp_3',
    'sp_4',
    'sp_5',
    'sp_6',
    'sp_7',
    'sp_8',
    'sp_9',
    'sp_10',
    'sp_11',
    'sp_12',
    'tp_1',
    'tp_2',
    'tp_3',
    'tp_4',
    'tp_5',
    'tp_6',
    'tp_7',
    'tp_8',
    'tp_9',
    'tp_10',
    'tp_11',
    'tp_12',
    'lai_hv_1',
    'lai_hv_2',
    'lai_hv_3',
    'lai_hv_4',
    'lai_hv_5',
    'lai_hv_6',
    'lai_hv_7',
    'lai_hv_8',
    'lai_hv_9',
    'lai_hv_10',
    'lai_hv_11',
    'lai_hv_12',
    'lai_lv_1',
    'lai_lv_2',
    'lai_lv_3',
    'lai_lv_4',
    'lai_lv_5',
    'lai_lv_6',
    'lai_lv_7',
    'lai_lv_8',
    'lai_lv_9',
    'lai_lv_10',
    'lai_lv_11',
    'lai_lv_12',
    'avg_elevation_m',
    'pct_wooded',
    'tick_pred',
    '5_to_14y_fraction',
    '50_to_65y_fraction',
    "roe_deer",
    "red_deer",
    "chinese_water_deer",
    "wood_mice_deer",
    "bank_voles_deer",
    "blackbird_deer",
]

# 2. read data
yaml_file_path = os.path.join("config", "variables.yaml")
variables = read_yaml(yaml_file_path)

X_train, y_train, X_val, y_val, X_test, df_test_year_areanm, N_train, N_val, N_test, D, df_train_meta_train, df_train_meta_val, scaler = load_train_test_data(seed, variables, count=True)


# 5. carve out kernel inputs vs linear inputs
#    - kernel on (year = col 0) and (space = cols 1–2)
#    - linear on everything else (cols 3:)
code_test_slice = 50 # np.inf
year_train   = X_train[:code_test_slice, 0]
X_space_train= X_train[:code_test_slice, 1:3]
X_lin_train  = X_train[:code_test_slice, 3:]
y_train = y_train[:code_test_slice].astype(int)

year_val     = X_val[:code_test_slice, 0]
X_space_val  = X_val[:code_test_slice, 1:3]
X_lin_val    = X_val[:code_test_slice, 3:]
y_val = y_val[:code_test_slice].astype(int)

year_test    = X_test[:code_test_slice, 0]
X_space_test = X_test[:code_test_slice, 1:3]
X_lin_test   = X_test[:code_test_slice, 3:]


N_train = X_space_train.shape[0]
N_val = X_space_val.shape[0]
N_test = X_space_test.shape[0]

stan_data = {
    'N_train':   N_train,
    'N_val':     N_val,
    'N_test':    N_test,
    'D_lin':     X_lin_train.shape[1],
    'y_train':   y_train,
    'y_val':     y_val,
    'year_train':   year_train,
    'X_space_train': X_space_train,
    'X_lin_train':   X_lin_train,
    'year_val':     year_val,
    'X_space_val':  X_space_val,
    'X_lin_val':    X_lin_val,
    'year_test':    year_test,
    'X_space_test': X_space_test,
    'X_lin_test':   X_lin_test,
}

model = CmdStanModel(stan_file="stan_models/gp.stan")
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_sampling=2000,
    iter_warmup=500,
)

# Get draws for y_hat (posterior predictive samples)
# ─── Extract posterior predictive draws for y_hat ───────────────────────────
# … after sampling …
posterior = fit.draws_pd()

# Extract only the training‐predictions columns
y_hat_cols    = [c for c in posterior.columns if c.startswith("y_hat_train")]
y_hat_samples = posterior[y_hat_cols].values  # (n_draws, N_train)

# Compute the mean while skipping NaNs
y_hat_mean = np.nanmean(y_hat_samples, axis=0)
# Compute the 95% CI bounds
y_hat_lower = np.nanpercentile(y_hat_samples, 2.5, axis=0)
y_hat_upper = np.nanpercentile(y_hat_samples, 97.5, axis=0)

# Mask out any NaN predictions
mask = ~np.isnan(y_hat_mean)
if not mask.all():
    n_bad = mask.size - mask.sum()
    print(f"Warning: dropping {n_bad} NaN predictions from RMSE")

# ─── Plot & save Stan‐GP predictions + CI ───────────────────────────────────
plot_predictions_with_ci(
    y_train,
    y_hat_mean,
    y_hat_lower,
    y_hat_upper,
    filepath=os.path.join(base_dir, "stan_gp_predictions.pdf"),
)

# ─── Compute & print RMSE on train/val ──────────────────────────────────────
rmse_val = np.sqrt(mean_squared_error(y_val[mask], y_hat_mean[mask]))

# ─── Build & sort your results dict ────────────────────────────────
# Compute means of each β using the real variable names:
lin_vars = variables[3:]   # drops 'year','latitude','longitude'
coef_means = {
    var: float(posterior[f"beta[{i+1}]"].mean())
    for i, var in enumerate(lin_vars)
}

# Sort coefficients by absolute magnitude, descending:
sorted_coefs = dict(
    sorted(coef_means.items(), key=lambda kv: kv[1], reverse=True)
)

results = {
    "validation_rmse": float(rmse_val),
    "coef_mean": sorted_coefs,
}

with open(os.path.join(base_dir, "results_gp_stan.json"), "w") as fp:
    json.dump(results, fp, indent=2)

# 1) Extract all β samples (they'll be named like beta[1], beta[2], …)
beta_cols = [c for c in posterior.columns if c.startswith("beta[")]
beta_samples = posterior[beta_cols]

# 2) Compute the posterior mean of each coefficient
coef_mean = beta_samples.mean(axis=0)

# 3) Build a 1-row DataFrame with human-readable column names
#    Assuming your `variables` list is in the same order and that
#    the first three cols of X were year, lat, lon, so betas start at index 3:
lin_vars = variables[3:]  
coef_df = pd.DataFrame([coef_mean.values], columns=lin_vars)

# 4) Plot & save
out_dir = f"results/GPR_hmc_{seed}"
os.makedirs(out_dir, exist_ok=True)
plot_coefficients(
    coef_df,
    "Coefficients",
    filepath=os.path.join(out_dir, "stan_gp_betas.pdf")
)