import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from xgboost import XGBClassifier
import itertools
import joblib

# File paths
FEATURES_FILE = "data/txs_features.csv"
LABELS_FILE = "data/txs_classes.csv"
ADDR_TX = "data/AddrTx_edgelist.csv"
TX_ADDR = "data/TxAddr_edgelist.csv"
WALLET_LABELS = "data/wallets_classes.csv"
MODEL_PATH = "VTAC/xgboost_model_10fold.pkl"

# VTAC selected features
MEANINGFUL_TX_FEATURES = [
    'in_txs_degree', 'out_txs_degree', 'total_BTC', 'fees', 'size',
    'num_input_addresses', 'num_output_addresses',
    'in_BTC_min', 'in_BTC_max', 'in_BTC_mean', 'in_BTC_median', 'in_BTC_total',
    'out_BTC_min', 'out_BTC_max', 'out_BTC_mean', 'out_BTC_median', 'out_BTC_total'
]

# Load pre-trained model
model = joblib.load(MODEL_PATH)
model_features = model.get_booster().feature_names

# Load address mappings
addr_tx_df = pd.read_csv(ADDR_TX).astype(str)
tx_addr_df = pd.read_csv(TX_ADDR).astype(str)

# Load wallet labels
wallet_labels_df = pd.read_csv(WALLET_LABELS)
wallet_labels_df["class"] = wallet_labels_df["class"].astype(int)
wallet_labels = dict(zip(wallet_labels_df["address"], wallet_labels_df["class"]))
wallets = list(wallet_labels.keys())


def load_data():
    df_features = pd.read_csv(FEATURES_FILE)
    df_labels = pd.read_csv(LABELS_FILE).rename(columns={"class": "label"})
    df_labels["label"] = df_labels["label"].astype(str).map({"1": 1, "2": 0, "unknown": -1})
    df = pd.merge(df_features, df_labels, on="txId", how="inner")
    df = df[df["label"] != -1].dropna()

    for feature in set(model_features) - set(df.columns):
        df[feature] = 0

    df = df[['txId', 'label'] + list(model_features)]
    return df


def get_predicted_illicit_wallets(tx_ids_pred):
    input_wallets = addr_tx_df[addr_tx_df["txId"].isin(tx_ids_pred)]["input_address"]
    output_wallets = tx_addr_df[tx_addr_df["txId"].isin(tx_ids_pred)]["output_address"]
    return set(input_wallets).union(set(output_wallets))


def evaluate_wallet_level(tx_ids_pred):
    predicted_illicit_wallets = get_predicted_illicit_wallets(tx_ids_pred)

    y_true = []
    y_pred = []

    for wallet, true_label in wallet_labels.items():
        if true_label not in [1, 2]:
            continue  # Skip unknown class (3)

        mapped_true = 1 if true_label == 1 else 0
        mapped_pred = 1 if wallet in predicted_illicit_wallets else 0

        y_true.append(mapped_true)
        y_pred.append(mapped_pred)

    return {
        "Precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "F1 Micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "F1 Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "AUC-PR": average_precision_score(y_true, y_pred)
    }


def print_metrics(metrics):
    print(f"Removed Feature(s): {metrics['Removed Feature']}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1 Micro: {metrics['F1 Micro']:.4f}")
    print(f"  F1 Macro: {metrics['F1 Macro']:.4f}")
    print(f"  AUC-PR: {metrics['AUC-PR']:.4f}\n")

# === MAIN ===
if __name__ == "__main__":
    while True:
        try:
            ablate_n = int(input(f"Choose number of features to remove (1-{len(MEANINGFUL_TX_FEATURES)}): "))
            if 1 <= ablate_n <= len(MEANINGFUL_TX_FEATURES):
                break
            else:
                print(f"Please enter a number between 1 and {len(MEANINGFUL_TX_FEATURES)}.")
        except ValueError:
            print("Please enter a valid integer.")

    df = load_data()
    X_full = df.drop(columns=["txId", "label"])
    y = df["label"].values
    tx_ids = df["txId"].astype(str).tolist()

    results = []

    # Baseline (all features)
    y_pred = model.predict(X_full.values)
    tx_ids_pred = [tid for tid, pred in zip(tx_ids, y_pred) if pred == 1]
    baseline = evaluate_wallet_level(tx_ids_pred)
    baseline["Removed Feature"] = "None"
    results.append(baseline)
    print("\nBaseline with all features:")
    print_metrics(baseline)

    # Feature ablation
    for features_to_remove in itertools.combinations(MEANINGFUL_TX_FEATURES, ablate_n):
        X_masked = X_full.copy()
        for feature in features_to_remove:
            if feature in X_masked.columns:
                X_masked[feature] = 0

        y_pred = model.predict(X_masked.values)
        tx_ids_pred = [tid for tid, pred in zip(tx_ids, y_pred) if pred == 1]
        metrics = evaluate_wallet_level(tx_ids_pred)
        metrics["Removed Feature"] = ', '.join(features_to_remove)
        results.append(metrics)
        print(f"\nAblated: {features_to_remove}")
        print_metrics(metrics)

    # Save results
    df_results = pd.DataFrame(results)
    output_file = f"vtac_wallet_ablation_{ablate_n}_removed_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nAblation results saved to {output_file}")
