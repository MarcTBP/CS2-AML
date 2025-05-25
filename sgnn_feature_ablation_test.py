"""
SGNN Feature Ablation Test Script (Last 17 Transaction Features)
---------------------------------------------------------------
Evaluates SGNN performance when each of the last 17 transaction features is dropped individually.
Assumes these are the meaningful features at the end of .x from txs_features_new.csv.
"""

import torch
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Step 1: Define the last 17 feature names (column C171–C187 from your CSV)
MEANINGFUL_TX_FEATURES = [
    "in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size",
    "num_input_addresses", "num_output_addresses",
    "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
    "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total"
]

# Step 2: Load the trained model
from SGNN.train.train_config import creat_SGNN_addr_att
model = creat_SGNN_addr_att()
checkpoint = torch.load(r'SGNN\results\addr\SGNN_addr_att\paras\now\SGNN.pth', map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()
print("Loaded SGNN model.")

# Step 3: Load data
from SGNN.dataloader.hetero_loader import get_hetero_train_test_loader
from SGNN.dataloader.hyper_loader import get_hyper_train_test_loader

rs_NP_ratio = 5
_, hetero_test_loader = get_hetero_train_test_loader(rs_NP_ratio)
_, hyper_test_loader = get_hyper_train_test_loader()
print("Loaded test data.")

# Step 4: Mask by index of the last N columns
def mask_transaction_feature_by_tail_index(loader, relative_index_from_end):
    masked_loader = []
    for data in loader:
        data_copy = deepcopy(data)
        total_cols = data_copy['transaction'].x.shape[1]
        col_to_zero = total_cols - 17 + relative_index_from_end
        data_copy['transaction'].x[:, col_to_zero] = 0
        masked_loader.append(data_copy)
    return masked_loader

# Step 5: Evaluation function
def evaluate_model(model, hetero_loader, hyper_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for h_data, hy_data in zip(hetero_loader, hyper_loader):
            out = model(hetero_data=h_data, hyper_data=hy_data)
            mask = h_data['address'].test_mask
            preds = out[mask].argmax(dim=1)
            probs = out[mask].softmax(dim=1)[:, 1]
            labels = h_data['address'].y[mask]
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob)
    }

# Step 6: Run ablation experiments
results = []

# Baseline with all features
print("Evaluating baseline (no features removed)...")
metrics = evaluate_model(model, hetero_test_loader, hyper_test_loader)
results.append({"Removed Feature": "None", **metrics})

# Loop to drop each of the last 17 features one by one
for i, feature_name in enumerate(MEANINGFUL_TX_FEATURES):
    print(f"Evaluating with feature removed: {feature_name}")
    masked_loader = mask_transaction_feature_by_tail_index(hetero_test_loader, i)
    metrics = evaluate_model(model, masked_loader, hyper_test_loader)
    results.append({"Removed Feature": feature_name, **metrics})

# Step 7: Save results and plot
df = pd.DataFrame(results)
df.to_csv("tail_feature_ablation_results.csv", index=False)
print("Results saved to tail_feature_ablation_results.csv")

plt.figure(figsize=(12, 6))
x_labels = df["Removed Feature"]
for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]:
    plt.plot(x_labels, df[metric], marker='o', label=metric)
plt.xlabel("Removed Feature (last 17 of transaction.x)")
plt.ylabel("Performance")
plt.title("SGNN Sensitivity to Last 17 Transaction Features")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tail_feature_ablation_plot.png")
plt.show()
