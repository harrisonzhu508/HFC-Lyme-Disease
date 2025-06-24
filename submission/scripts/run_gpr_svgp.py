# Load data 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from utils import read_yaml, load_train_test_data, plot_coefficients, plot_predictions_with_ci
import os
import gpflow
import numpy as np
from gpflow.kernels import Matern32, RBF
from gpflow.mean_functions import Constant
from gpflow.config import default_float
import tensorflow as tf
import gpflow
import tensorflow as tf
from gpflow.inducing_variables import InducingPoints
from tqdm import tqdm

# Custom Negative Binomial Likelihood
class NegativeBinomialLikelihood(gpflow.likelihoods.Likelihood):
    def __init__(self, input_dim=1, latent_dim=1, observation_dim=1, phi=1.0):
        super().__init__(input_dim, latent_dim, observation_dim)
        # phi must be positive
        self.phi = gpflow.Parameter(phi, transform=gpflow.utilities.positive())

    def _log_prob(self, F, Y):
        # Cast to ensure consistency
        phi = tf.cast(self.phi, F.dtype)
        F = tf.cast(F, F.dtype)
        Y = tf.cast(Y, F.dtype)
        
        log_prob = (tf.math.lgamma(Y + phi)
                    - tf.math.lgamma(Y + 1.0)
                    - tf.math.lgamma(phi)
                    + phi * tf.math.log(phi)
                    + Y * F
                    - (Y + phi) * tf.math.log(tf.math.exp(F) + phi))
        return log_prob

    def _predict_log_density(self, F, Y):
        return self._log_prob(F, Y)

    def _variational_expectations(self, X, Fmu, Fvar, Y):
        quad_order = 20
        nodes, weights = np.polynomial.hermite.hermgauss(quad_order)
        nodes = tf.convert_to_tensor(nodes, dtype=default_float())
        weights = tf.convert_to_tensor(weights, dtype=default_float())
        Fmu = tf.squeeze(Fmu, axis=-1)
        Fvar = tf.squeeze(Fvar, axis=-1)
        Y = tf.squeeze(Y, axis=-1)
        # Evaluate F at quadrature nodes
        F = tf.sqrt(2.0 * Fvar)[None, :] * nodes[:, None] + Fmu[None, :]
        log_probs = self._log_prob(F, Y[None, :])
        return tf.reduce_sum(weights[:, None] * log_probs, axis=0) / np.sqrt(np.pi)

    def _predict_mean_and_var(self, Fmu, Fvar):
        mu = tf.exp(Fmu + 0.5 * Fvar)
        exp2F = tf.exp(2 * Fmu + 2 * Fvar)
        var_expF = tf.exp(2 * Fmu + Fvar) * (tf.exp(Fvar) - 1)
        var = mu + exp2F / self.phi + var_expF
        return mu, var

import argparse
parser = argparse.ArgumentParser(description="Run Lasso regression with bootstrapping.")
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
args = parser.parse_args()
seed = args.seed
os.makedirs(f"results/GPR_svgp_{seed}", exist_ok=True)

#####################
##### LOAD DATA #####
#####################
yaml_file_path = os.path.join("config", "variables.yaml")
variables = read_yaml(yaml_file_path)

X_train, y_train, X_val, y_val, X_test, df_test_year_areanm, N_train, N_val, N_test, D, df_train_meta_train, df_train_meta_val, scaler = load_train_test_data(seed, variables, count=True)

# Simulate data structure
X_tensor = tf.convert_to_tensor(X_train, dtype=default_float())
Y_tensor = tf.convert_to_tensor(y_train, dtype=default_float())[:, None]
X_val_tensor = tf.convert_to_tensor(X_val, dtype=default_float())
D = X_train.shape[1]
N_train = len(y_train)
N_val = len(y_val)
print(f"Training set size: {N_train}, Validation set size: {N_val}, Test set size: {N_test}, Number of predictors: {D}, y_train shape: {y_train.shape}, X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}, y_val shape: {y_val.shape}")

# Build kernel components
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



# Additive kernel
kernel = (
    kernel_latlon*kernel_year +
    kernel_t2m +
    kernel_spec_humidity +
    kernel_rel_humidity +
    kernel_src +
    kernel_sp +
    kernel_tp +
    kernel_lai_hv +
    kernel_lai_lv +
    kernel_avg_elevation +
    kernel_pct_wooded +
    kernel_tick_pred + 
    kernel_age + 
    kernel_roe_deer +
    kernel_red_deer +
    kernel_chinese_water_deer +
    kernel_wood_mice_deer +
    kernel_bank_voles_deer +
    kernel_blackbird_deer
)


# Select inducing points (e.g., M randomly chosen points)
M = min(100, N_train)
# use kmeans to select Z
# Z = X_tensor[:M].numpy().astype(np.float64)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=M, random_state=seed)
kmeans.fit(X_train)
Z = kmeans.cluster_centers_.astype(np.float64)

# Create the SVGP model
# Create the SVGP model
model = gpflow.models.SVGP(
    kernel=kernel,
    likelihood=NegativeBinomialLikelihood(phi=1.0),  # Ensure this is defined/imported
    inducing_variable=InducingPoints(Z),
    num_latent_gps=1,
    mean_function=Constant(0.0)
)

# Parameters
batch_size = N_train
num_steps = 20000

# Prepare minibatched dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_tensor, Y_tensor))
train_dataset = train_dataset.shuffle(buffer_size=1000).repeat().batch(batch_size)
train_iter = iter(train_dataset)

# Optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.01)

@tf.function
def optimization_step(X_batch, Y_batch):
    with tf.GradientTape() as tape:
        loss = model.training_loss((X_batch, Y_batch))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# # Optimization loop with minibatches
# for step in tqdm(range(num_steps)):
#     X_batch, Y_batch = next(train_iter)
#     loss = optimization_step(X_batch, Y_batch)

#     if step % 500 == 0:
#         print(f"Step {step}: Loss = {loss.numpy()}")

# Save the model
model.compiled_predict_f = tf.function(
    lambda Xnew: model.predict_f(Xnew, full_cov=False),
    input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
)

log_dir = f"results/GPR_svgp_{seed}/checkpoints"
save_dir = f"results/GPR_svgp_{seed}/gpflow_model"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
ckpt = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=5)
# Save the model
# manager.save()
if manager.latest_checkpoint:
    print(f"Restoring from {manager.latest_checkpoint}")
    ckpt.restore(manager.latest_checkpoint)
else:
    print("Initializing from scratch")

# load model:
# model = tf.saved_model.load(model_file_path)

print("SVGP optimization with minibatch SGD completed.")

# 95% CI on counts
# 1) Get latent predictive mean & variance
Fmu, Fvar = model.predict_f(X_val_tensor)    # both are [N,1] tensors

# 2) Convert to count space via your NB likelihood
mu_counts, var_counts = model.likelihood._predict_mean_and_var(Fmu, Fvar)
# 3) Build 95% CIs on counts
std_counts      = tf.sqrt(var_counts)
ci_lower_counts = mu_counts - 1.96 * std_counts
ci_upper_counts = mu_counts + 1.96 * std_counts

# 4) Convert to incidence per 100k
den = tf.constant(df_train_meta_val["Denominator"].values,
                  dtype=mu_counts.dtype)[:, None]
scale = 100_000

mean_incidence     = mu_counts       / den * scale
ci_lower_incidence = ci_lower_counts / den * scale
ci_upper_incidence = ci_upper_counts / den * scale
y_true_incidence   = tf.cast(y_val[:, None], mu_counts.dtype) / den * scale

import pandas as pd
import matplotlib.pyplot as plt

# Convert tensors to NumPy
mean_np = mean_incidence.numpy().flatten()
ci_lower_np = ci_lower_incidence.numpy().flatten()
ci_upper_np = ci_upper_incidence.numpy().flatten()
y_true_np = y_true_incidence.numpy().flatten()

# print out shapes 
print(f"Mean shape: {mean_np.shape}")
print(f"CI Lower shape: {ci_lower_np.shape}")
print(f"CI Upper shape: {ci_upper_np.shape}")
print(f"True values shape: {y_true_np.shape}")
# Print the first few values
print("Mean predictions (first 5):", mean_np[:5])
print("CI Lower (first 5):", ci_lower_np[:5])
print("CI Upper (first 5):", ci_upper_np[:5])

# Plotting
plot_predictions_with_ci(y_true_np, mean_np, ci_lower_np, ci_upper_np, filepath=f"results/GPR_svgp_{seed}/gp_predictions.pdf")
# print RMSE 
from sklearn.metrics import mean_squared_error
rmse_gp = np.sqrt(mean_squared_error(y_true_np, mean_np))
print(f"Validation RMSE (GPFlow): {rmse_gp:.4f}")




import matplotlib.pyplot as plt
# Extract kernel scales and true feature names from "variables" variable: variables

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
    "blackbird_deer"
]

# 1. Extract & sort your kernel variances
kernel_scales = {
    gp_variables[i]: float(model.kernel.kernels[i].variance.numpy())
    for i in range(1, len(model.kernel.kernels))
}
kernel_scales["latlon_year"] = (
    model.kernel.kernels[0].kernels[0].variance.numpy()
    * model.kernel.kernels[0].kernels[1].variance.numpy()
)

kernel_scales_sorted = dict(
    sorted(kernel_scales.items(), key=lambda kv: kv[1], reverse=True)
)

# 2. Make a 1-row wide df
coef_wide = pd.DataFrame([kernel_scales_sorted])

# 3. Ensure the column order matches your sort
coef_wide = coef_wide[list(kernel_scales_sorted.keys())]

coef_wide = coef_wide[kernel_scales_sorted.keys()]

plot_coefficients(
    coef_wide,
    "Kernel Scales",
    os.path.join("results", f"GPR_svgp_{seed}", "gp_kernel_scales.pdf")
)

import json
results_gp = {
    "validation_rmse": rmse_gp,
    "kernel_scales": kernel_scales_sorted
}
results_gp_file_path = os.path.join("results", f"GPR_svgp_{seed}", "gpflow_results.json")
os.makedirs(os.path.dirname(results_gp_file_path), exist_ok=True)

with open(results_gp_file_path, 'w') as f:
    json.dump(results_gp, f, indent=4)  


# save test predictions as CSV file with column names Year,Council,Incidence,Lower_95CI,Upper_95CI
# 1) Latent predictions
mean, var = model.predict_f(
    tf.convert_to_tensor(X_test, dtype=default_float())
)  # both are [N,1] tensors

# 2) Transform to count‐space via the NB likelihood
mu_counts, var_counts = model.likelihood._predict_mean_and_var(mean, var)

# 3) Build 95% CIs on counts
std_counts      = tf.sqrt(var_counts)
ci_lower_counts = mu_counts - 1.96 * std_counts
ci_upper_counts = mu_counts + 1.96 * std_counts

# 4) Convert to incidence per 100k
den   = tf.constant(df_test_year_areanm["Denominator"].values,
                    dtype=mu_counts.dtype)[:, None]
scale = 100_000

# **Define** these before using them below
mean_incidence     = mu_counts       / den * scale   # [N,1]
ci_lower_incidence = ci_lower_counts / den * scale   # [N,1]
ci_upper_incidence = ci_upper_counts / den * scale   # [N,1]

# 5) pull out NumPy arrays
test_preds     = mean_incidence.numpy().flatten()
test_ci_lower  = ci_lower_incidence.numpy().flatten()
test_ci_upper  = ci_upper_incidence.numpy().flatten()

test_summary = pd.DataFrame({
    "Council": df_test_year_areanm["areanm"],
    "Year": df_test_year_areanm["year"],
    'Incidence': test_preds,
    'Lower_95CI': test_ci_lower,
    'Upper_95CI': test_ci_upper,
})
test_summary_file_path = os.path.join("results", f"GPR_svgp_{seed}", "test_predictions.csv")
os.makedirs(os.path.dirname(test_summary_file_path), exist_ok=True)
test_summary.to_csv(test_summary_file_path, index=False)

# save validation predictions as CSV file with column names Year,Council,Incidence,Lower_95CI,Upper_95CI
val_summary = pd.DataFrame({
    "Council": df_train_meta_val["areanm"],
    "Year": df_train_meta_val["year"],
    'Incidence': mean_np,
    'Lower_95CI': ci_lower_np,
    'Upper_95CI': ci_upper_np,
})
val_summary_file_path = os.path.join("results", f"GPR_svgp_{seed}", "val_predictions.csv")
os.makedirs(os.path.dirname(val_summary_file_path), exist_ok=True)
val_summary.to_csv(val_summary_file_path, index=False)


def compute_ale_incidence_nb_sampling_joint(
    model,
    X,               # (N,D) numpy array
    denom,           # (N,) numpy array
    feature_idx,
    n_bins=20,
    scale=100_000,
    n_samples=200,
    random_state=0,
    jitter=1e-6,
):
    rng = np.random.RandomState(random_state)
    phi = float(model.likelihood.phi.numpy())  # dispersion

    # bin edges & centers
    xj = X[:, feature_idx]
    edges = np.linspace(xj.min(), xj.max(), n_bins+1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # store per-bin samples of Δ_k
    delta_samps = np.zeros((n_bins, n_samples))

    for k in range(n_bins):
        # include right‐edge on last bin
        if k == n_bins-1:
            mask = (xj >= edges[k]) & (xj <= edges[k+1])
        else:
            mask = (xj >= edges[k]) & (xj <  edges[k+1])
        n_k = mask.sum()
        if n_k == 0:
            continue

        # build low/high inputs
        X_low  = X[mask].copy()
        X_high = X[mask].copy()
        X_low[:,  feature_idx] = edges[k]
        X_high[:, feature_idx] = edges[k+1]
        den = denom[mask]  # shape (n_k,)

        # 1) Stack inputs and get joint posterior of latent f
        X_both = np.vstack([X_low, X_high])            # (2*n_k, D)
        Fmu, Fcov = model.predict_f(X_both, full_cov=True)
        Fmu = Fmu.numpy().flatten()                    # (2*n_k,)
        Fcov = Fcov.numpy()                            # (2*n_k,2*n_k)

        # 2) Draw joint samples of f
        L = np.linalg.cholesky(Fcov + jitter * np.eye(2*n_k))
        normals = rng.randn(2*n_k, n_samples)          # (2*n_k, S)
        f_samps = Fmu[:, None] + L @ normals           # (2*n_k, S)
        f_samps = f_samps.squeeze(axis=0)  # (2*n_k, S)

        # split low/high samples
        f_low_samps  = f_samps[:n_k, :]                # (n_k, S)
        f_high_samps = f_samps[n_k:, :]                # (n_k, S)

        # 3) Convert to NB counts via Gamma–Poisson
        mu_low_samps  = np.exp(f_low_samps)
        mu_high_samps = np.exp(f_high_samps)
        lam_low  = rng.gamma(shape=phi, scale=(mu_low_samps/phi))
        lam_high = rng.gamma(shape=phi, scale=(mu_high_samps/phi))
        cnt_low_samps  = rng.poisson(lam_low)
        cnt_high_samps = rng.poisson(lam_high)

        # 4) To incidence + scale
        m_low_samps  = (cnt_low_samps  / den[:, None]) * scale  # (n_k, S)
        m_high_samps = (cnt_high_samps / den[:, None]) * scale

        # 5) Per-sample bin-Δ_k
        delta_samps[k, :] = (m_high_samps - m_low_samps).mean(axis=0)

    # 6) Accumulate + center each sample’s ALE
    ale_samps = np.cumsum(delta_samps, axis=0)          # (n_bins, S)
    ale_samps -= ale_samps.mean(axis=0)[None, :]        # center

    # 7) Summarize
    ale_mean = ale_samps.mean(axis=1)                   # (n_bins,)
    ale_se   = ale_samps.std(axis=1, ddof=1)            # (n_bins,)

    # ale_samps: shape (n_bins, n_samples)
    abs_samps = np.abs(ale_samps)                 # (n_bins, S)
    imp_samples = abs_samps.mean(axis=0)          # (S,)  — average abs ALE per draw

    imp_mean = imp_samples.mean()
    imp_ci   = np.percentile(imp_samples, [2.5, 97.5])

    return centers, ale_mean, ale_se, imp_mean, imp_ci




def plot_ale_delta_method(
    centers: np.ndarray,
    ale_mean: np.ndarray,
    ale_se:   np.ndarray,
    feature_values: np.ndarray,
    feature_name:    str,
    out_path:        str,
    ci_multiplier: float = 1.96,
):
    """
    Plot ALE mean +/- ci_multiplier * se with rug ticks.
    """
    # Compute bounds
    lower = ale_mean - ci_multiplier * ale_se
    upper = ale_mean + ci_multiplier * ale_se

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(centers, lower, upper,
                    alpha=0.3, label=f"{100*(2*(1 - 0.975)):.0f}% CI")
    ax.plot(centers, ale_mean, lw=2, label="ALE mean")

    # Labels and title
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Incidence‐ALE (per 100k)")
    ax.set_title(f"ALE for feature {feature_name}")
    ax.legend()

    # Rug plot under x‐axis
    y_min, y_max = ax.get_ylim()
    rug_y = y_min - 0.05 * (y_max - y_min)
    rug_h = 0.03 * (y_max - y_min)
    ax.eventplot(feature_values,
                 lineoffsets=rug_y,
                 linelengths=rug_h,
                 colors='k', alpha=0.5)

    # Save and close
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# === usage ===
pbar = tqdm(range(len(variables)), desc="Looping")
for i in pbar:
    pbar.set_postfix(variable=variables[i])
    centers, ale_mean, ale_se, imp_mean, imp_ci = compute_ale_incidence_nb_sampling_joint(
        model,
        X_train,
        denom=df_train_meta_train["Denominator"].values,
        feature_idx=i,
        n_bins=40,
        scale=100_000,
    )
    out_file = os.path.join(
        "results",
        f"GPR_svgp_{seed}",
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

    plot_ale_delta_method(
        centers_orig,
        ale_mean,
        ale_se,
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
        f"GPR_svgp_{seed}",
        f"ale_data_feature_{variables[i]}.json"
    )
    with open(ale_data_file, "w") as f:
        json.dump(ale_data, f, indent=4)

    
