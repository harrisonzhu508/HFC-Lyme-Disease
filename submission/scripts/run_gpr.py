# run_gpflow_shap.py
"""
Gaussian Process Regression (GPflow) with existing kernel design, bootstrap‑style
predictive intervals from GP variance, **plus SHAP importance** via kernel
explainer.

We do **not** change model fitting or prediction logic—only append SHAP analysis
and dump mean |SHAP| values to JSON.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import json
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import gpflow
from gpflow.kernels import Matern32, RBF
from gpflow.mean_functions import Constant
from gpflow.models import GPR
from gpflow.config import default_float
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utils import read_yaml, load_train_test_data, plot_coefficients, plot_predictions_with_ci
from tqdm import tqdm
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run GP regression with SHAP.")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
args = parser.parse_args()
seed = args.seed
np.random.seed(seed)
os.makedirs(f"results/GPR_{seed}", exist_ok=True)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
yaml_file_path = os.path.join("config", "variables.yaml")
variables = read_yaml(yaml_file_path)
(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    df_test_year_areanm,
    *_,
    df_train_meta_train,
    df_train_meta_val,
    scaler
) = load_train_test_data(seed, variables)

# Convert to tensors
X_tensor = tf.convert_to_tensor(X_train, dtype=default_float())
Y_tensor = tf.convert_to_tensor(y_train[:, None], dtype=default_float())
X_val_tensor = tf.convert_to_tensor(X_val, dtype=default_float())

# -----------------------------------------------------------------------------
# Kernel definition (same as user script)
# -----------------------------------------------------------------------------
# Build kernel components (omitted repeated imports)
kernel_year = RBF(active_dims=[0])
kernel_latlon = Matern32(active_dims=[1, 2])
kernel_t2m = RBF(active_dims=list(range(3, 15)))
kernel_spec_humidity = RBF(active_dims=list(range(15, 27)))
kernel_rel_humidity = RBF(active_dims=list(range(27, 39)))
kernel_src = RBF(active_dims=list(range(39, 51)))
kernel_sp = RBF(active_dims=list(range(51, 63)))
kernel_tp = RBF(active_dims=list(range(63, 75)))
kernel_lai_hv = RBF(active_dims=list(range(75, 87)))
kernel_lai_lv = RBF(active_dims=list(range(87, 99)))
kernel_avg_elevation = RBF(active_dims=[99])
kernel_pct_wooded = RBF(active_dims=[100])
kernel_tick_pred = RBF(active_dims=[101])
kernel_age = RBF(active_dims=[102, 103])
kernel_roe_deer = RBF(active_dims=[103])
kernel_red_deer = RBF(active_dims=[104])
kernel_chinese_water_deer = RBF(active_dims=[105])
kernel_wood_mice_deer = RBF(active_dims=[106])
kernel_bank_voles_deer = RBF(active_dims=[107])
kernel_blackbird_deer = RBF(active_dims=[108])

kernel = (
    kernel_latlon * kernel_year
    + kernel_t2m
    + kernel_spec_humidity
    + kernel_rel_humidity
    + kernel_src
    + kernel_sp
    + kernel_tp
    + kernel_lai_hv
    + kernel_lai_lv
    + kernel_avg_elevation
    + kernel_pct_wooded
    + kernel_tick_pred
    + kernel_age
    + kernel_roe_deer
    + kernel_red_deer
    + kernel_chinese_water_deer
    + kernel_wood_mice_deer
    + kernel_bank_voles_deer
    + kernel_blackbird_deer
)

# -----------------------------------------------------------------------------
# Fit GP model
# -----------------------------------------------------------------------------
model = GPR(data=(X_tensor, Y_tensor), kernel=kernel, mean_function=Constant(), noise_variance=1e-5)
opt = gpflow.optimizers.Scipy()
# opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100), method="L-BFGS-B")


log_dir = f"results/GPR_{seed}/checkpoints"
save_dir = f"results/GPR_{seed}/gpflow_model"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
ckpt = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=5)
# manager.save()
if manager.latest_checkpoint:
    print(f"Restoring from {manager.latest_checkpoint}")
    ckpt.restore(manager.latest_checkpoint)
else:
    print("Initializing from scratch")

# Load the model if needed
# model = tf.saved_model.load(save_dir)

# -----------------------------------------------------------------------------
# Predictions and intervals
# -----------------------------------------------------------------------------
mean, var = model.predict_y(X_val_tensor)
ci_lower = mean - 1.96 * tf.sqrt(var)
ci_upper = mean + 1.96 * tf.sqrt(var)
mean_np, ci_lower_np, ci_upper_np = mean.numpy().flatten(), ci_lower.numpy().flatten(), ci_upper.numpy().flatten()
rmse_gp = np.sqrt(mean_squared_error(y_val, mean_np))
print(f"Validation RMSE (GPFlow): {rmse_gp:.4f}")

plot_predictions_with_ci(
    y_val.flatten(), mean_np, ci_lower_np, ci_upper_np, filepath=f"results/GPR_{seed}/gp_predictions.pdf"
)

# -----------------------------------------------------------------------------
# Kernel scale feature importance (unchanged)
# -----------------------------------------------------------------------------
gp_variables = [
    "latlon_year",
    "t2m",
    "spec_humidity",
    "rel_humidity",
    "src",
    "sp",
    "tp",
    "lai_hv",
    "lai_lv",
    "avg_elevation",
    "pct_wooded",
    "tick_pred",
    "age",
    "roe_deer",
    "red_deer",
    "chinese_water_deer",
    "wood_mice_deer",
    "bank_voles_deer",
    "blackbird_deer",
]

kernel_scales = {gp_variables[i]: float(model.kernel.kernels[i + 1].variance.numpy()) for i in range(len(gp_variables) - 1)}
# latlon_year special (product)
kernel_scales["latlon_year"] = model.kernel.kernels[0].kernels[0].variance.numpy() * model.kernel.kernels[0].kernels[1].variance.numpy()

kernel_scales_sorted = dict(sorted(kernel_scales.items(), key=lambda kv: kv[1], reverse=True))
coef_wide = pd.DataFrame([kernel_scales_sorted])[kernel_scales_sorted.keys()]
plot_coefficients(coef_wide, "Kernel Scales", os.path.join("results", f"GPR_{seed}", "gp_kernel_scales.pdf"))

# -----------------------------------------------------------------------------
# SHAP via KernelExplainer (slow but generic)
# -----------------------------------------------------------------------------
# Use a subset of training points as background
# background_size = min(100, X_train.shape[0])
# background = X_train[np.random.choice(X_train.shape[0], background_size, replace=False)]

# # Prediction function that returns 1‑D numpy array (mean)

# def gp_predict(X_):
#     mu, _ = model.predict_f(tf.convert_to_tensor(X_, dtype=default_float()))
#     return mu.numpy().flatten()

# explainer = shap.KernelExplainer(gp_predict, background)
# # Compute SHAP on validation set (can limit for speed)
# shap_values = explainer.shap_values(X_val, nsamples=20)  # nsamples controls Monte‑Carlo size
# shap_imp = np.abs(shap_values).mean(axis=0)
# shap_ordered = (
#     pd.Series(shap_imp, index=variables)
#     .sort_values(ascending=False)
#     .to_dict()
# )

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
results_gp = {
    "validation_rmse": rmse_gp,
    "kernel_scales": kernel_scales_sorted,
    # "shap_importance": shap_ordered,
}
results_gp_file = os.path.join("results", f"GPR_{seed}", "gpflow_results.json")
os.makedirs(os.path.dirname(results_gp_file), exist_ok=True)
with open(results_gp_file, "w") as f:
    json.dump(results_gp, f, indent=4)

# ---------------- Save predictions to CSV ----------------
mean_test, var_test = model.predict_y(tf.convert_to_tensor(X_test, dtype=default_float()))
mean_test_np = mean_test.numpy().flatten()
ci_test_lower = mean_test_np - 1.96 * np.sqrt(var_test.numpy().flatten())
ci_test_upper = mean_test_np + 1.96 * np.sqrt(var_test.numpy().flatten())

test_summary = pd.DataFrame(
    {
        "Council": df_test_year_areanm["areanm"],
        "Year": df_test_year_areanm["year"],
        "Incidence": mean_test_np,
        "Lower_95CI": ci_test_lower,
        "Upper_95CI": ci_test_upper,
    }
)
val_summary = pd.DataFrame(
    {
        "Council": df_train_meta_val["areanm"],
        "Year": df_train_meta_val["year"],
        "Incidence": mean_np,
        "Lower_95CI": ci_lower_np,
        "Upper_95CI": ci_upper_np,
    }
)

test_summary.to_csv(os.path.join("results", f"GPR_{seed}", "test_predictions.csv"), index=False)
val_summary.to_csv(os.path.join("results", f"GPR_{seed}", "val_predictions.csv"), index=False)



# ─── 4) Plot ALE for “other” feature j=3 ────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def compute_ale_gaussian_sampling(
    model,
    X,                # (N,D) numpy array
    feature_idx: int, 
    n_bins: int = 40,
    n_samples: int = 200,
    random_state: int = 0,
    jitter: float = 1e-6,
):
    """
    Monte‐Carlo ALE for a GPR with Gaussian noise.
    Returns bin centers, mean ALE, and pointwise ALE‐SE.
    """
    rng = np.random.RandomState(random_state)
    noise_var = float(model.likelihood.variance.numpy())

    # 1) set up bins
    xj    = X[:, feature_idx]
    edges = np.linspace(xj.min(), xj.max(), n_bins+1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 2) placeholder for per-bin Δ samples
    delta_samps = np.zeros((n_bins, n_samples))

    for k in range(n_bins):
        # mask for bin k (include right edge on last bin)
        if k == n_bins-1:
            mask = (xj >= edges[k]) & (xj <= edges[k+1])
        else:
            mask = (xj >= edges[k]) & (xj <  edges[k+1])
        n_k = mask.sum()
        if n_k == 0:
            continue

        # build “low” & “high” inputs
        X_low  = X[mask].copy()
        X_high = X[mask].copy()
        X_low[:,  feature_idx] = edges[k]
        X_high[:, feature_idx] = edges[k+1]

        # 3) joint posterior cov of predictive y = f + ε
        X_both = np.vstack([X_low, X_high])            # (2*n_k, D)
        Fmu, Fcov = model.predict_f(X_both, full_cov=True)
        Fmu = Fmu.numpy().flatten()                    # (2*n_k,)
        Fcov = Fcov.numpy()                            # (2*n_k,2*n_k)

        # add noise variance on the diagonal to get predictive-cov(Y)
        Fcov += noise_var * np.eye(2*n_k)

        # 4) draw S joint samples of predictive y
        L = np.linalg.cholesky(Fcov + jitter * np.eye(2*n_k))
        normals = rng.randn(2*n_k, n_samples)          # standard normals
        y_samps = Fmu[:, None] + L @ normals           # (2*n_k, S)
        y_samps = y_samps.squeeze(0)

        # split into low/high and compute Δ samples
        y_low  = y_samps[:n_k, :]                      # (n_k, S)
        y_high = y_samps[n_k:, :]                      # (n_k, S)
        delta_samps[k, :] = (y_high - y_low).mean(axis=0)

    # 5) build full ALE curves and center each draw
    ale_samps = np.cumsum(delta_samps, axis=0)         # (n_bins, S)
    ale_samps -= ale_samps.mean(axis=0)[None, :]       # center

    # 6) posterior summaries
    ale_mean = ale_samps.mean(axis=1)                  # (n_bins,)
    ale_se   = ale_samps.std(axis=1, ddof=1)           # (n_bins,)
    # after you’ve built ale_samps of shape (n_bins, S):
    
    # ale_samps: shape (n_bins, n_samples)
    abs_samps = np.abs(ale_samps)                 # (n_bins, S)
    imp_samples = abs_samps.mean(axis=0)          # (S,)  — average abs ALE per draw

    imp_mean = imp_samples.mean()
    imp_ci   = np.percentile(imp_samples, [2.5, 97.5])

    return centers, ale_mean, ale_se, imp_mean, imp_ci


def plot_ale_gaussian(
    centers: np.ndarray,
    ale_mean: np.ndarray,
    ale_se:   np.ndarray,
    feature_values: np.ndarray,
    feature_name:    str,
    out_path:        str,
    ci: tuple = (1.96,),
):
    """
    Plot ALE mean +/- ci*se with rug.
    """
    lower = ale_mean - ci[0] * ale_se
    upper = ale_mean + ci[0] * ale_se
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(centers, lower, upper, alpha=0.3, label="95% CI")
    ax.plot(centers, ale_mean, lw=2, label="ALE mean")
    
    ax.set_xlabel(feature_name)
    ax.set_ylabel("ALE")
    ax.set_title(f"ALE for feature {feature_name}")
    ax.legend()
    
    # rug
    y_min, y_max = ax.get_ylim()
    rug_y = y_min - 0.05*(y_max-y_min)
    rug_h = 0.03*(y_max-y_min)
    ax.eventplot(feature_values, lineoffsets=rug_y,
                 linelengths=rug_h, colors='k', alpha=0.5)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Usage in loop
pbar = tqdm(range(len(variables)), desc="Looping")
for i in pbar:
    pbar.set_postfix(variable=variables[i])
    centers, ale_mean, ale_se, imp_mean, imp_ci = compute_ale_gaussian_sampling(
        model, X_train, feature_idx=i, n_bins=40
    )
    out_file = os.path.join(
        "results",
        f"GPR_{seed}",
        f"ale_feature_{variables[i]}.pdf"
    )

    # scale back to original feature values
    if scaler is not None:
        # scaler scales all features, but we only need the i-th feature from sklearn scale
        # To invert the x‐axis for feature i:
        mean_i  = scaler.mean_[i]
        scale_i = scaler.scale_[i]

        # centers is in scaled‐units → back to original
        centers_orig = centers * scale_i + mean_i

        X_train_original = X_train[:, i] * scale_i + mean_i

    else:
        raise ValueError("Scaler is None, cannot inverse transform centers, ale_mean, ale_se.")

    plot_ale_gaussian(
        centers_orig, ale_mean, ale_se,
        X_train_original,
        variables[i],
        out_file
    )

    # save the ALE data
    ale_data = {
        "centers": centers_orig.tolist(),
        "ale_mean": ale_mean.tolist(),
        "ale_se": ale_se.tolist(),
        "feature_name": variables[i],
        "feature_values": X_train_original.tolist(),
        "ale_overall_mean": imp_mean,
        "ale_lo": imp_ci[0],
        "ale_hi": imp_ci[1]
    }
    ale_data_file = os.path.join(
        "results",
        f"GPR_{seed}",
        f"ale_data_feature_{variables[i]}.json"
    )
    with open(ale_data_file, "w") as f:
        json.dump(ale_data, f, indent=4)
