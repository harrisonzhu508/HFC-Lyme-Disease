# run_lasso_bootstrap_shap.py
"""
Lasso regression with **bootstrapped prediction intervals** *and* SHAP
feature‑importance. All existing bootstrap logic is preserved; only SHAP
computation and JSON export are added.

Usage
-----
    python run_lasso_bootstrap_shap.py --seed 0

Outputs under `results/lasso_<seed>/`:
    • val_predictions.csv / test_predictions.csv
    • coefficients.pdf (bootstrap means)
    • lasso_results.json  (includes SHAP importances)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

from utils import load_train_test_data, plot_coefficients, read_yaml

# ----------------------------- CLI -------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Lasso regression with bootstrapping and SHAP analysis."
)
parser.add_argument("--seed", type=int, default=0, help="Random seed")
args = parser.parse_args()
seed = args.seed
np.random.seed(seed)

# ----------------------------- LOAD DATA -------------------------------------
variables_yaml = os.path.join("config", "variables.yaml")
feature_names = read_yaml(variables_yaml)
(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    df_test_year_areanm,
    N_train,
    N_val,
    N_test,
    D,
    df_train_meta_train,
    df_train_meta_val,
    scaler
) = load_train_test_data(seed, feature_names)

# ----------------------------- MODEL + SHAP ----------------------------------
lasso_alpha = 0.1
n_bootstraps = 1000

lasso = Lasso(alpha=lasso_alpha, random_state=seed)
lasso.fit(X_train, y_train)

# SHAP on *fitted* model (training data)
explainer = shap.LinearExplainer(lasso, X_train)
shap_values = explainer.shap_values(X_train)
shap_imp = np.abs(shap_values).mean(axis=0)
shap_ordered = (
    pd.Series(shap_imp, index=feature_names)
    .sort_values(ascending=False)
    .to_dict()
)

# ----------------------------- BOOTSTRAP -------------------------------------
val_preds = np.zeros((n_bootstraps, X_val.shape[0]))
test_preds = np.zeros((n_bootstraps, X_test.shape[0]))
coef_samples = np.zeros((n_bootstraps, X_train.shape[1]))

for i in range(n_bootstraps):
    X_res, y_res = resample(X_train, y_train, replace=True, random_state=seed + i)
    model = Lasso(alpha=lasso_alpha, random_state=seed + i)
    model.fit(X_res, y_res)
    val_preds[i] = model.predict(X_val)
    test_preds[i] = model.predict(X_test)
    coef_samples[i] = model.coef_

val_ci_lower = np.percentile(val_preds, 2.5, axis=0)
val_ci_upper = np.percentile(val_preds, 97.5, axis=0)
val_pred_mean = val_preds.mean(axis=0)

test_ci_lower = np.percentile(test_preds, 2.5, axis=0)
test_ci_upper = np.percentile(test_preds, 97.5, axis=0)
test_pred_mean = test_preds.mean(axis=0)

# ----------------------------- SUMMARIES -------------------------------------
val_summary = pd.DataFrame(
    {
        "Council": df_train_meta_val["areanm"],
        "Year": df_train_meta_val["year"],
        "Incidence": val_pred_mean,
        "Lower_95CI": val_ci_lower,
        "Upper_95CI": val_ci_upper,
    }
)

test_summary = pd.DataFrame(
    {
        "Council": df_test_year_areanm["areanm"],
        "Year": df_test_year_areanm["year"],
        "Incidence": test_pred_mean,
        "Lower_95CI": test_ci_lower,
        "Upper_95CI": test_ci_upper,
    }
)

print("Validation predictions (head):\n", val_summary.head())
print("\nTest predictions (head):\n", test_summary.head())

# ----------------------------- COEFFICIENT PLOT ------------------------------
coef_df = pd.DataFrame(coef_samples, columns=feature_names)
coef_df = coef_df.reindex(coef_df.mean().sort_values(ascending=False).index, axis=1)

results_dir = os.path.join("results", f"lasso_{seed}")
os.makedirs(results_dir, exist_ok=True)
plot_coefficients(
    coef_df,
    "Coefficients (Bootstrap Means)",
    os.path.join(results_dir, "coefficients.pdf"),
)

# ----------------------------- METRICS & SAVE -------------------------------
rmse_val = np.sqrt(mean_squared_error(y_val, val_pred_mean))
print(f"Validation RMSE (mean bootstrap prediction): {rmse_val:.4f}")

results_json = {
    "validation_rmse": rmse_val,
    "feature_importance": coef_df.mean().to_dict(),
    "shap_importance": shap_ordered,
}
with open(os.path.join(results_dir, "lasso_results.json"), "w") as f:
    json.dump(results_json, f, indent=4)

# Save predictions
val_summary.to_csv(os.path.join(results_dir, "val_predictions.csv"), index=False)
test_summary.to_csv(os.path.join(results_dir, "test_predictions.csv"), index=False)
