# Load data 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def read_yaml(file_path: str) -> dict:
    """Read a YAML file and return its contents as a dictionary."""
    with open(file_path, 'r') as file:
        variables = yaml.safe_load(file)
    variables = [var for var in variables if var not in ["areanm", "Incidence", "Lower CI 95.0 limit", "Upper CI 95.0 limit", "Lower CI 99.8 limit", "Upper CI 99.8 limit", "country", "count", "Denominator", "mean_tree_cover_pct", "tick_std"]]
    return variables

def load_train_test_data(seed, variables, count=False):
    """Load training and test data from CSV files."""

    df_train = pd.read_csv("data/processed_data/train.csv")
    df_test = pd.read_csv("data/processed_data/test.csv")
    if count:
        y_train = df_train["count"].values
    else:
        y_train = df_train["Incidence"].values
    # y_train = df_train["count"].values
    # Cast y_train to int
    # y_train = y_train.astype(np.int32)  # Or round first if needed: np.round(y_train).astype(np.int32)

    # mean_denominator = df_train.groupby("areanm")["Denominator"].mean().reset_index()
    df_train_year_areanm = df_train[["areanm", "year", "Denominator"]]
    
    df_train = df_train[["areanm"] + variables]
    # df_train = pd.merge(df_train, mean_denominator, on="areanm", how="left")
    # df_train = df_train[["Denominator"] + variables]
    df_train = df_train[variables]
    X_train = df_train.values


    # grab all rows for year==2022, then only keep the first row per area
    denoms_2022 = (
        df_train_year_areanm[df_train_year_areanm["year"] == 2022]
        .drop_duplicates(subset="areanm", keep="first")
        .set_index("areanm")["Denominator"]
    )

    # map it onto your test set
    df_test["Denominator"] = df_test["areanm"].map(denoms_2022)
    
    df_test_year_areanm = df_test[["areanm", "year", "Denominator"]]
    N_test = len(df_test)
    # df_test = df_test[["Denominator"] + variables]
    df_test = df_test[variables]
    X_test = df_test.values

    N = len(y_train)
    D = X_train.shape[1]  # Number of predictors

    # split X, y, AND your metadata all at once
    X_train, X_val, \
    y_train, y_val, \
    df_train_meta_train, df_train_meta_val = train_test_split(
        X_train,
        y_train,
        df_train_year_areanm,
        test_size=0.2,
        random_state=seed,
    )

    # Define new dimensions
    N_train = len(y_train)  # Size of new training set
    N_val = len(y_val)          # Size of validation set

    # apply scaling to X_train, X_val, and X_test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Print dimensions for confirmation
    print(f"Original training size: {N}")
    print(f"New training size: {N_train}")
    print(f"Validation size: {N_val}")
    print(f"Test size: {N_test}")
    print(f"Number of predictors (D): {D}")

    return X_train, y_train, X_val, y_val, X_test, df_test_year_areanm, N_train, N_val, N_test, D, df_train_meta_train, df_train_meta_val, scaler


def plot_predictions_with_ci(y_true, y_pred, ci_lower, ci_upper, filepath):
    plt.figure(figsize=(10, 6))

    # Calculate error bars (ensuring they are positive)
    yerr_lower = np.abs(y_pred - ci_lower)
    yerr_upper = np.abs(ci_upper - y_pred)
    xerr = [yerr_lower, yerr_upper]

    # Error bars first for layering effect
    plt.errorbar(
        y_pred, y_true, xerr=xerr,
        fmt='o', ecolor='lightgray', elinewidth=1, capsize=3,
        alpha=0.6, label='95% CI'
    )

    # Scatter points (on top)
    plt.scatter(
        y_pred, y_true, color='steelblue', edgecolor='white',
        alpha=0.8, s=60, label='Predictions'
    )

    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot(
        [min_val, max_val], [min_val, max_val],
        'k--', linewidth=1.5, label='Perfect Prediction'
    )

    # Labels and title
    plt.title('Observed vs. Predicted with 95% Confidence Intervals', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Observed', fontsize=12)

    # Formatting
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    plt.show()
    plt.savefig(filepath, bbox_inches='tight')

def plot_coefficients(coef_df, name, filepath):
    """
    Inputs:
        - coef_df: DataFrame containing coefficients from bootstraps
        - filepath: Path to save the plot
    Outputs:
        - A violin plot showing the distribution of coefficients across bootstraps
    1. Saves the plot to the specified filepath.
    2. Displays the plot.
    3. Coefficients are ordered by their mean value.
    4. Uses seaborn for visualization.
    5. The plot is saved with tight layout to avoid clipping of labels.
    6. The figure size is set to (25, 6) for better visibility.
    """
    # # Create a DataFrame for coefficients
    # coef_df = pd.DataFrame(coef_samples, columns=feature_names)
    
    # # Order coefficients by mean value
    # coef_df = coef_df.reindex(coef_df.mean().sort_values(ascending=False).index, axis=1)
    
    # Melt for seaborn
    coef_long = coef_df.melt(var_name='Feature', value_name=name)
    
    plt.figure(figsize=(25, 6))
    sns.violinplot(data=coef_long, x='Feature', y=name, inner='box')
    plt.xticks(rotation=45)
    # plt.title('Bootstrap Distributions of Lasso Coefficients')
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
    plt.savefig(filepath, bbox_inches='tight')