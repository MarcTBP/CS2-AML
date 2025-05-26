import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import sys
import itertools
import joblib

# File paths
FEATURES_FILE = "data/txs_features.csv"
LABELS_FILE = "data/txs_classes.csv"

# Load the pre-trained model from Our_VTAC.py
model = joblib.load("xgboost_model_10fold.pkl")

# Get the feature names from the model
model_features = model.get_booster().feature_names

# 17 meaningful features selected for ablation testing
MEANINGFUL_TX_FEATURES = [
    'in_txs_degree', 'out_txs_degree', 'total_BTC', 'fees', 'size',
    'num_input_addresses', 'num_output_addresses',
    'in_BTC_min', 'in_BTC_max', 'in_BTC_mean', 'in_BTC_median', 'in_BTC_total',
    'out_BTC_min', 'out_BTC_max', 'out_BTC_mean', 'out_BTC_median', 'out_BTC_total'
]


def load_data():
    df_features = pd.read_csv(FEATURES_FILE)
    df_labels = pd.read_csv(LABELS_FILE).rename(columns={"class": "label"})
    label_map = {"1": 1, "2": 0, "unknown": -1}  # Updated to match Our_VTAC.py mapping
    df_labels["label"] = df_labels["label"].astype(str).map(label_map)

    df = pd.merge(df_features, df_labels, on="txId", how="inner")
    df = df[df["label"] != -1].dropna()

    # Ensure we have all the features the model expects
    missing_features = set(model_features) - set(df.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with zeros
        for feature in missing_features:
            df[feature] = 0

    # Reorder columns to match model's expected feature order
    df = df[['txId', 'label'] + list(model_features)]

    return df


def evaluate_full_model(X, y):
    # Use the pre-trained model to make predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Micro": f1_score(y, y_pred, average="micro"),
        "F1 Macro": f1_score(y, y_pred, average="macro"),
        "AUC-PR": average_precision_score(y, y_proba)
    }


def print_metrics(metrics):
    print(f"Removed Feature(s): {metrics['Removed Feature']}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1 Micro: {metrics['F1 Micro']:.4f}")
    print(f"  F1 Macro: {metrics['F1 Macro']:.4f}")
    print(f"  AUC-PR: {metrics['AUC-PR']:.4f}\n")


# === Main execution ===
if __name__ == "__main__":
    # Prompt user for number of features to ablate
    while True:
        try:
            ablate_n = int(input(f"Choose number of features to remove (1-{len(MEANINGFUL_TX_FEATURES)}): "))
            if 1 <= ablate_n <= len(MEANINGFUL_TX_FEATURES):
                break
            else:
                print(f"Please enter a number between 1 and {len(MEANINGFUL_TX_FEATURES)}.")
        except ValueError:
            print("Please enter a valid integer.")

    print(f"Loading dataset and running baseline evaluation on full dataset...")
    df = load_data()
    X_full = df.drop(columns=["txId", "label"])
    y = df["label"].values

    results = []

    # Baseline
    print("\nEvaluating baseline model with all features...")
    baseline = evaluate_full_model(X_full.values, y)
    baseline["Removed Feature"] = "None"
    results.append(baseline)
    print_metrics(baseline)

    # Feature ablation loop
    print(f"\nPerforming ablation study by removing {ablate_n} feature(s) at a time...")
    for features_to_remove in itertools.combinations(MEANINGFUL_TX_FEATURES, ablate_n):
        print(f"\nEvaluating without feature(s): {features_to_remove}")
        # Create a copy of X_full with zeros for removed features
        X_subset = X_full.copy()
        for feature in features_to_remove:
            if feature in X_subset.columns:
                X_subset[feature] = 0

        metrics = evaluate_full_model(X_subset.values, y)
        metrics["Removed Feature"] = ', '.join(features_to_remove)
        results.append(metrics)
        print_metrics(metrics)

    # Save results
    df_results = pd.DataFrame(results)
    output_file = f"vtac_feature_ablation_{ablate_n}_removed_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nAblation results saved to {output_file}")

    # Print summary of results
    print("\nSummary of ablation study:")
    print("=" * 50)
    print(f"Number of features removed: {ablate_n}")
    print(f"Total combinations tested: {len(results) - 1}")  # -1 for baseline
    print("\nTop 5 worst performing feature combinations:")
    worst_combinations = df_results.sort_values('F1 Macro').head(6)  # Include baseline
    print(worst_combinations[['Removed Feature', 'F1 Macro', 'Precision', 'Recall']])