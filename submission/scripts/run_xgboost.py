# run_xgboost_conformal.py
"""
Run XGBoost with **split‑conformal prediction** (no bootstrapping) and SHAP feature
importance.

Usage
-----
    python run_xgboost_conformal.py --seed 42

This writes predictions + conformal 95 % intervals, feature‑importance PDF, and
metrics JSON to
    results/xgboost_<seed>/

Conformal method
----------------
1. **Train / calibration split**: we split the original training data (X_train)
   into an inner train set and a calibration set.
2. Fit one XGBRegressor on the inner train set.
3. Compute absolute residuals on the calibration set; the (1 – α) empirical
   quantile (α = 0.05 → 95 % PI) becomes the half‑width *q* of every interval.
4. For any new point, prediction interval is `ŷ ± q`.
This gives finite‑sample, distribution‑free coverage (assuming i.i.d. data).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from PyALE import ale as ale_fun  # avoid name clash with module
from scipy.stats import norm
from tqdm import tqdm
from utils import load_train_test_data, plot_coefficients, read_yaml
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def plot_ale_gaussian(
    centers, ale_mean, ci_lo, ci_hi,
    feature_values,
    feature_name,
    out_path,
):
    # compute bounds
    lower = ci_lo
    upper = ci_hi

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(centers, lower, upper, alpha=0.3, label="95 % CI")
    ax.plot(centers, ale_mean, lw=2, label="ALE mean")
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

def train_model(X_tr, y_tr, seed, n_estimators=100, learning_rate=0.1):
    """Fit and return an XGBRegressor."""
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)
    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(seed: int, calib_frac: float = 0.2, alpha: float = 0.05):
    np.random.seed(seed)

    # ---------------- Load data -----------------
    feature_yaml = os.path.join("config", "variables.yaml")
    feature_names = read_yaml(feature_yaml)

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

    # ------------ Split train ↔ calibration ----
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=calib_frac, random_state=seed
    )

    # --------------- Fit model -----------------
    model = train_model(X_train, y_train, seed)

    # --------------- SHAP importance -----------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_imp = (
        np.abs(shap_values).mean(axis=0)
    )  # mean(|SHAP|) per feature
    shap_ordered = (
        pd.Series(shap_imp, index=feature_names)
        .sort_values(ascending=False)
        .to_dict()
    )

    # --------------- Conformal calibration -----
    cal_resid = np.abs(y_cal - model.predict(X_cal))
    q_hat = np.quantile(cal_resid, 1 - alpha)  # half‑width

    def predict_interval(X):
        y_hat = model.predict(X)
        lower = y_hat - q_hat
        upper = y_hat + q_hat
        return y_hat, lower, upper

    # Intervals for validation & test
    val_pred_mean, val_ci_lower, val_ci_upper = predict_interval(X_val)
    test_pred_mean, test_ci_lower, test_ci_upper = predict_interval(X_test)

    # ---------------- Summaries ----------------
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

    # --------------- Feature importance PDF ----
    fi_df = pd.DataFrame([model.feature_importances_], columns=feature_names)
    fi_df = fi_df.reindex(fi_df.mean().sort_values(ascending=False).index, axis=1)

    results_dir = os.path.join("results", f"xgboost_{seed}")
    os.makedirs(results_dir, exist_ok=True)
    plot_coefficients(
        fi_df,
        "Feature Importance (Single‑fit)",
        os.path.join(results_dir, "feature_importance.pdf"),
    )

    # --------------- Metrics & save ------------
    rmse_val = np.sqrt(mean_squared_error(y_val, val_pred_mean))
    print(f"Validation RMSE: {rmse_val:.4f}; conformal q̂ = {q_hat:.4f}")

    results = {
        "validation_rmse": rmse_val,
        "conformal_half_width": float(q_hat),
        "feature_importance": fi_df.mean().to_dict(),
        "shap_importance": shap_ordered,
    }
    with open(os.path.join(results_dir, "xgboost_results.json"), "w") as f:
        json.dump(results, f, indent=4)

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
            "ale_overall_mean": float(np.mean(ale_mean)),
            "ale_lo": float(np.percentile(ale_mean, 5)),
            "ale_hi": float(np.percentile(ale_mean, 95)),
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
        description="Run XGBoost regression with conformal prediction and SHAP analysis."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    main(args.seed)
