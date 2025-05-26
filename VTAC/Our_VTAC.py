import pandas as pd
import networkx as nx
import os
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, average_precision_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load data
features = pd.read_csv("txs_features.csv")
classes = pd.read_csv("txs_classes.csv").rename(columns={"class": "label"})
edges = pd.read_csv("txs_edgelist.csv")

# Map labels
label_map = {"1": 1, "2": 0, "unknown": -1}
classes["label"] = classes["label"].astype(str).map(label_map)

# Merge features and labels
data = pd.merge(features, classes, on="txId", how="left")
id_to_label = dict(zip(data["txId"], data["label"]))

# Build graph
G = nx.from_pandas_edgelist(edges, source="txId1", target="txId2", create_using=nx.DiGraph())

# Visual representation of the graph
sample_nodes = list(G.nodes)[:500]  # Adjust this number if needed
G_sub = G.subgraph(sample_nodes).copy()
color_map = []
for node in G_sub.nodes():
    label = id_to_label.get(node, -1)
    if label == 1:
        color_map.append("red")    # Illicit
    elif label == 0:
        color_map.append("green")  # Licit
    else:
        color_map.append("black")  # Unknown

pos = nx.spring_layout(G_sub, seed=42)
plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G_sub, pos, node_color=color_map, node_size=20)
nx.draw_networkx_edges(G_sub, pos, alpha=0.3, width=0.2)
plt.title("Subgraph: Red=Illicit, Green=Licit, Black=Unknown")
plt.axis("off")
plt.tight_layout()
plt.show()

# Add graph-based features
deg_df = pd.DataFrame({
    "txId": list(G.nodes),
    "in_deg": [G.in_degree(n) for n in G.nodes],
    "out_deg": [G.out_degree(n) for n in G.nodes],
    "pagerank": pd.Series(nx.pagerank(G))
})
data = pd.merge(data, deg_df, on="txId", how="left").fillna(0)

# Filter labeled data
labeled_data = data[data["label"] != -1]
X = labeled_data.drop(columns=["txId", "label"])
X.columns = X.columns.astype(str)
y = labeled_data["label"]
tx_ids = labeled_data["txId"].values

# Initialize metrics
metrics_list = {
    "Precision": [], "Recall": [], "F1 (Micro)": [],
    "F1 (Macro)": [], "AUC-PR": []
}
all_y_test, all_y_pred, all_y_proba = [], [], []
all_tx_ids_test, all_tx_ids_pred = [], []

# 10-Fold Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    tx_test_fold = tx_ids[test_idx]

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Store results
    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)
    all_tx_ids_test.extend(tx_test_fold)
    all_tx_ids_pred.extend(tx_test_fold[y_pred == 1])

    # Compute metrics
    metrics_list["Precision"].append(precision_score(y_test, y_pred))
    metrics_list["Recall"].append(recall_score(y_test, y_pred))
    metrics_list["F1 (Micro)"].append(f1_score(y_test, y_pred, average='micro'))
    metrics_list["F1 (Macro)"].append(f1_score(y_test, y_pred, average='macro'))
    metrics_list["AUC-PR"].append(average_precision_score(y_test, y_proba))

# Final evaluation
print("\nOverall Classification Report:")
print(classification_report(all_y_test, all_y_pred, target_names=["Licit", "Illicit"]))

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(all_y_test, all_y_proba)
auc_pr = average_precision_score(all_y_test, all_y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f'AP = {auc_pr:.4f}', color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Average metric plot
avg_metrics = {k: np.mean(v) for k, v in metrics_list.items()}

plt.figure(figsize=(8, 6))
plt.bar(avg_metrics.keys(), avg_metrics.values(), color='orange')
plt.ylabel("Score")
plt.title("Average Evaluation Metrics (10-Fold CV)")
plt.ylim(0, 1.05)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Save model (last fold's)
joblib.dump(model, "xgboost_model_10fold.pkl")

# Save predicted illicit transactions and edges
bad_tx_ids = pd.Series(all_tx_ids_pred).drop_duplicates()
bad_tx_ids.to_csv("Predicted_IllicitTx_txID.csv", index=False)

bad_edges = edges[
    edges["txId1"].isin(bad_tx_ids) | edges["txId2"].isin(bad_tx_ids)
].copy()

bad_edges["illicit_node"] = (
    edges["txId1"].isin(bad_tx_ids).astype(int) +
    2 * edges["txId2"].isin(bad_tx_ids).astype(int)
)

# Address mapping
input_address = pd.read_csv("AddrTx_edgelist.csv")
output_address = pd.read_csv("TxAddr_edgelist.csv")
address_to_tx = dict(zip(input_address["txId"], input_address["input_address"]))
tx_to_address = dict(zip(output_address["txId"], output_address["output_address"]))

bad_edges["address_txId1"] = bad_edges["txId1"].map(address_to_tx)
bad_edges["address_txId2"] = bad_edges["txId2"].map(tx_to_address)

bad_edges.to_csv("Predicted_IllicitEdges_txID.csv", index=False)

print(f"Number of predicted illicit transactions: {len(bad_tx_ids)}")
print("Saving to:", os.path.abspath("Predicted_IllicitTx_Edges.csv"))
print(f"bad_edges shape: {bad_edges.shape}")

# Feature importance from last model
plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=15)
plt.title("Top 15 Most Important Features (XGBoost - Last Fold)")
plt.tight_layout()
plt.show()