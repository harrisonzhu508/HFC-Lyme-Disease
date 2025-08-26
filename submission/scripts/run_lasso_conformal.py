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
import matplotlib.pyplot as plt
from PyALE import ale as ale_fun  # avoid name clash with module
from scipy.stats import norm
from tqdm import tqdm

from utils import load_train_test_data, plot_coefficients, read_yaml

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------
## ALE Curves
def plot_ale_gaussian(
    centers, ale_mean, ci_lo, ci_hi,
    feature_values,
    feature_name,
    out_path,
):
    # compute bounds
    lower = ci_lo
    upper = ci_hi

    # for the leftmost point, for lower and upper, we make them equal to the mean

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, ale_mean, lw=2, label="ALE mean")
    ax.fill_between(centers, lower, upper, alpha=0.3, label="95 % CI")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("ALE")
    ax.set_title(f"ALE for feature {feature_name}")
    ax.legend()

    # rug
    y_min, y_max = ax.get_ylim()
    rug_y = y_min - 0.05 * (y_max - y_min)
    rug_h = 0.03 * (y_max - y_min)
    ax.eventplot(feature_values,
                 lineoffsets=rug_y,
                 linelengths=rug_h,
                 colors="k",
                 alpha=0.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    
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


    # -----------------------------------------------------------------
    print("\nComputing ALE curves …")
    print(feature_names)
    print(X_train.shape, len(feature_names))
    X_df = pd.DataFrame(X_train, columns=feature_names)

    ale_dir = os.path.join(results_dir, "ale_curves")
    os.makedirs(ale_dir, exist_ok=True)
    ale_importance = {}          # mean(abs(ALE)) per feature
    ale_curve_store = {}         # raw curves → JSON
    
    pbar = tqdm(range(len(feature_names)), desc="Looping")
    for i in pbar:
        feat = feature_names[i]
        pbar.set_postfix(variable=feature_names[i])
        out_pdf = os.path.join(ale_dir, f"ale_feature_{feat}.pdf")

        # compute ALE with built-in CI (grid_size instead of nbins)
        ale_res = ale_fun(
            X=X_df,
            model=model,
            feature=[feat],     # PyALE expects a list for 1D
            include_CI=True,
            C=0.95,             # 95% interval
            grid_size=40        # number of bins
        )

        # extract bin centers & ALE mean
        centers  = ale_res.index.values
        ale_mean = ale_res["eff"].values

        # extract PyALE’s CIs
        ci_lo = ale_res["lowerCI_95%"].values
        ci_hi = ale_res["upperCI_95%"].values

        # convert to “se” so plot_ale_gaussian can do ±1.96·se
        z        = norm.ppf(0.975)               # ~1.96
        # ci_lo = np.nan_to_num(ci_lo, nan=0.0)      # <- **new**: replace NaN with 0
        # ci_hi = np.nan_to_num(ci_hi, nan=0.0)      # <- **new**: replace NaN with 0
        ale_se   = (ci_hi - ci_lo) / (2 * z)

        # original feature values for the rug
        feat_vals = X_df[feat].values

        # invert scaling if applicable
        if scaler is not None:
            idx       = feature_names.index(feat)
            centers   = centers   * scaler.scale_[idx] + scaler.mean_[idx]
            feat_vals = feat_vals * scaler.scale_[idx] + scaler.mean_[idx]

        # plot
        plot_ale_gaussian(
            centers,
            ale_mean,
            ci_lo,
            ci_hi,
            feat_vals,
            feat,
            out_pdf
        )

        # save to dicts
        ale_importance[feat] = np.mean(np.abs(ale_mean))
        ale_curve_store[feat] = {
            "centers":   centers.tolist(),
            "ale_mean":  ale_mean.tolist(),
            "ale_se":    ale_se.tolist(),
            "ci_lower":  ci_lo.tolist(),
            "ci_upper":  ci_hi.tolist(),
            "ale_overall_mean": np.mean(np.abs(ale_mean)),
            "ale_lo": np.percentile(ale_mean, 5).item(),
            "ale_hi": np.percentile(ale_mean, 95).item(),
        }

    print("ALE curves saved to", ale_dir)
    # save raw ALE curves for downstream use
    with open(os.path.join(results_dir, "ale_curves.json"), "w") as f:
        json.dump(ale_curve_store, f, indent=4)

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
