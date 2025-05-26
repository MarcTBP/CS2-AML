import pandas as pd
import matplotlib.pyplot as plt
import os

# === Config ===
SGNN_FILE = "tail_feature_ablation_results.csv"
VTAC_FILE = "vtac_wallet_ablation_1_removed_results.csv"
OUTPUT_DIR = "comparison_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

metrics = ["Precision", "Recall", "F1 Micro", "F1 Macro", "AUC-PR"]

# === Load Data ===
sgnn_df = pd.read_csv(SGNN_FILE)
vtac_df = pd.read_csv(VTAC_FILE)

# Clean and align
sgnn_df = sgnn_df[sgnn_df["Removed Feature"].notna()]
vtac_df = vtac_df[vtac_df["Removed Feature"].notna()]

sgnn_df["Removed Feature"] = sgnn_df["Removed Feature"].astype(str)
vtac_df["Removed Feature"] = vtac_df["Removed Feature"].astype(str)

# Align feature order (assumes both datasets use the same 17 features)
common_features = [f for f in sgnn_df["Removed Feature"] if f != "None"]

# Filter to common features only
sgnn_plot = sgnn_df[sgnn_df["Removed Feature"].isin(common_features)]
vtac_plot = vtac_df[vtac_df["Removed Feature"].isin(common_features)]

# Ensure same order
sgnn_plot = sgnn_plot.set_index("Removed Feature").loc[common_features]
vtac_plot = vtac_plot.set_index("Removed Feature").loc[common_features]

# === Plot for each metric ===
for metric in metrics:
    plt.figure(figsize=(14, 6))
    x = range(len(common_features))
    plt.plot(x, sgnn_plot[metric], marker='o', label="SGNN")
    plt.plot(x, vtac_plot[metric], marker='s', label="VTAC")

    # Add labels
    for i, feat in enumerate(common_features):
        plt.text(i, sgnn_plot[metric][i] + 0.005, f"{sgnn_plot[metric][i]:.3f}", ha="center", fontsize=7)
        plt.text(i, vtac_plot[metric][i] - 0.05, f"{vtac_plot[metric][i]:.3f}", ha="center", fontsize=7)

    plt.xticks(x, common_features, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.xlabel("Removed Feature")
    plt.ylabel("Score")
    plt.title(f"{metric} Comparison: SGNN vs VTAC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{metric.replace(' ', '_')}_comparison.png")
    plt.close()

print("Comparison plots saved to:", OUTPUT_DIR)
