# run_lasso_conformal.py
"""
Run Lasso regression with **split‑conformal prediction** (no bootstrapping)
plus SHAP feature‐importance.

Usage
-----
    python run_lasso_conformal.py --seed 42

Outputs (predictions with 95 % PI, feature‑importance PDF, JSON metrics) land in
    results/lasso_<seed>/
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # keep GPU env in case of remote clusters

import argparse
import json

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from utils import load_train_test_data, plot_coefficients, read_yaml

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main(seed: int, calib_frac: float = 0.2, alpha_conf: float = 0.05, lasso_alpha: float = 0.1):
    np.random.seed(seed)

    # -------- Load variables & data ---------
    variables_yaml = os.path.join("config", "variables.yaml")
    feature_names = read_yaml(variables_yaml)

    (
        X_train_full,
        y_train_full,
        X_val,
        y_val,
        X_test,
        df_test_year_areanm,
        *_,
        df_train_meta_train,
        df_train_meta_val,
        scaler
    ) = load_train_test_data(seed, feature_names)

    # ----- Train/Calibration split ----------
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=calib_frac, random_state=seed
    )

    # -------------- Fit model ---------------
    model = Lasso(alpha=lasso_alpha, random_state=seed)
    model.fit(X_train, y_train)

    # -------------- SHAP values -------------
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    shap_imp = np.abs(shap_values).mean(axis=0)
    shap_ordered = (
        pd.Series(shap_imp, index=feature_names)
        .sort_values(ascending=False)
        .to_dict()
    )

    # ------ Split‑conformal half‑width ------
    cal_residuals = np.abs(y_cal - model.predict(X_cal))
    q_hat = np.quantile(cal_residuals, 1 - alpha_conf)  # 95 % PI if alpha_conf=0.05

    def predict_interval(X):
        y_hat = model.predict(X)
        return y_hat, y_hat - q_hat, y_hat + q_hat

    # ---------- Validation & Test -----------
    val_pred, val_lower, val_upper = predict_interval(X_val)
    test_pred, test_lower, test_upper = predict_interval(X_test)

    # ----------- DataFrames -----------------
    val_summary = pd.DataFrame(
        {
            "Council": df_train_meta_val["areanm"],
            "Year": df_train_meta_val["year"],
            "Incidence": val_pred,
            "Lower_95CI": val_lower,
            "Upper_95CI": val_upper,
        }
    )
    test_summary = pd.DataFrame(
        {
            "Council": df_test_year_areanm["areanm"],
            "Year": df_test_year_areanm["year"],
            "Incidence": test_pred,
            "Lower_95CI": test_lower,
            "Upper_95CI": test_upper,
        }
    )

    print("Validation predictions (head):\n", val_summary.head())
    print("\nTest predictions (head):\n", test_summary.head())

    # ----------- Feature Importance PDF ------
    coef_df = pd.DataFrame([model.coef_], columns=feature_names)
    coef_df = coef_df.reindex(coef_df.iloc[0].abs().sort_values(ascending=False).index, axis=1)

    results_dir = os.path.join("results", f"lasso_conformal_{seed}")
    os.makedirs(results_dir, exist_ok=True)
    plot_coefficients(
        coef_df,
        "Lasso Coefficients",
        os.path.join(results_dir, "coefficients.pdf"),
    )

    # -------------- Metrics & Save ----------
    rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {rmse_val:.4f}; conformal q̂ = {q_hat:.4f}")

    results_json = {
        "validation_rmse": rmse_val,
        "conformal_half_width": float(q_hat),
        "feature_importance": coef_df.iloc[0].to_dict(),
        "shap_importance": shap_ordered,
    }

    with open(os.path.join(results_dir, "lasso_results.json"), "w") as f:
        json.dump(results_json, f, indent=4)

    # Save predictions CSV
    val_summary.to_csv(os.path.join(results_dir, "val_predictions.csv"), index=False)
    test_summary.to_csv(os.path.join(results_dir, "test_predictions.csv"), index=False)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Lasso regression with conformal prediction and SHAP analysis."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    main(args.seed)
