import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
CSV_FILE = "vtac_wallet_ablation_1_removed_results.csv"  # or "vtac_wallet_ablation_1_removed_results.csv"
TITLE = "Feature Ablation Impact on SGNN Performance"
XLABEL = "Removed Feature"
OUTPUT_IMAGE = "feature_ablation_full_plot.png"

# === Load Data ===
df = pd.read_csv(CSV_FILE)
df = df[df["Removed Feature"].notna()]  # clean
df["Removed Feature"] = df["Removed Feature"].astype(str)

# === Plot All Metrics with Value Labels ===
plt.figure(figsize=(14, 6))
x_labels = df["Removed Feature"]
metrics = ["Precision", "Recall", "F1 Micro", "F1 Macro", "AUC-PR"]

for metric in metrics:
    plt.plot(x_labels, df[metric], marker='o', label=metric)
    for x, y in zip(x_labels, df[metric]):
        plt.text(x, y + 0.005, f"{y:.3f}", ha='center', va='bottom', fontsize=7)

plt.xlabel(XLABEL)
plt.ylabel("Score")
plt.title(TITLE)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE)
plt.show()
