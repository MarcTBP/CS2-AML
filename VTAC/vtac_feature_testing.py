import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import sys
import itertools

# File paths
FEATURES_FILE = "txs_features.csv"
LABELS_FILE = "txs_classes.csv"

# 17 meaningful features selected for ablation testing
MEANINGFUL_TX_FEATURES = [
    'in_txs_degree', 'out_txs_degree', 'total_BTC', 'fees', 'size',
    'num_input_addresses', 'num_output_addresses',
    'in_BTC_min', 'in_BTC_max', 'in_BTC_mean', 'in_BTC_median', 'in_BTC_total',
    'out_BTC_min', 'out_BTC_max', 'out_BTC_mean', 'out_BTC_median', 'out_BTC_total'
]


class VTACModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(VTACModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))


def load_data():
    df_features = pd.read_csv(FEATURES_FILE)
    df_labels = pd.read_csv(LABELS_FILE).rename(columns={"class": "label"})
    label_map = {"1": 0, "2": 1, "3": -1}
    df_labels["label"] = df_labels["label"].astype(str).map(label_map)

    df = pd.merge(df_features, df_labels, on="txId", how="inner")
    df = df[df["label"] != -1].dropna()

    return df


def evaluate_full_model(X, y):
    class_counts = pd.Series(y).value_counts()
    print("Class distribution before resampling:", dict(class_counts))

    apply_smote = class_counts.min() > 5
    if apply_smote:
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
        print("Applied SMOTE.")
    else:
        print("Skipping SMOTE: Not enough minority class samples.")

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    model = VTACModel(input_dim=X_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor.unsqueeze(1))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = model(X_tensor).numpy()
        preds = (probs > 0.5).astype(float)

    return {
        "Precision": precision_score(y_tensor, preds),
        "Recall": recall_score(y_tensor, preds),
        "F1 Micro": f1_score(y_tensor, preds, average="micro"),
        "F1 Macro": f1_score(y_tensor, preds, average="macro"),
        "AUC-PR": average_precision_score(y_tensor, probs)
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
    baseline = evaluate_full_model(X_full.values, y)
    baseline["Removed Feature"] = "None"
    results.append(baseline)
    print_metrics(baseline)

    # Feature ablation loop (all combinations of ablate_n features)
    for features_to_remove in itertools.combinations(MEANINGFUL_TX_FEATURES, ablate_n):
        print(f"Evaluating without feature(s): {features_to_remove}")
        X_subset = X_full.drop(columns=list(features_to_remove)).values
        metrics = evaluate_full_model(X_subset, y)
        metrics["Removed Feature"] = ', '.join(features_to_remove)
        results.append(metrics)
        print_metrics(metrics)

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"vtac_feature_ablation_{ablate_n}_removed_results.csv", index=False)
    print(f"Ablation results saved to vtac_feature_ablation_{ablate_n}_removed_results.csv")
