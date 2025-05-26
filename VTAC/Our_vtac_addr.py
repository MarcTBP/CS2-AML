import pandas as pd #for data handling
import os #used for paths to csv files
import networkx as nx #for graph creation/analysis
from sklearn.model_selection import train_test_split #for data splitting and evaluation
from imblearn.over_sampling import SMOTE #for balancing classes using SMOTE
from xgboost import XGBClassifier, plot_importance #training model and for feature visualization
from sklearn.metrics import classification_report, confusion_matrix #Evaluate predicions
import matplotlib.pyplot as plt #for creating visualizations

#Link to dataset: https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l

base_dir = os.path.dirname(__file__)  # Gets the directory of the script

# Load only the first 10,000 rows (or fewer if you want faster)
START_ROW = 0
SAMPLE_ROWS = 20000

addr_addr = pd.read_csv(
    os.path.join(base_dir, 'data/AddrAddr_edgelist.csv'),
    skiprows=range(1, START_ROW),
    nrows=SAMPLE_ROWS
)

input_map = pd.read_csv(
    os.path.join(base_dir, 'data/AddrTx_edgelist.csv'),
    skiprows=range(1, START_ROW),
    nrows=SAMPLE_ROWS
)

output_map = pd.read_csv(
    os.path.join(base_dir, 'data/TxAddr_edgelist.csv'),
    skiprows=range(1, START_ROW),
    nrows=SAMPLE_ROWS
)

# Merge to replace input_address with its txId
addr_with_input_id = addr_addr.merge(input_map, left_on='input_address', right_on='input_address', how='left')
addr_with_input_id = addr_with_input_id.drop(columns=['input_address', 'input_address'])
addr_with_input_id = addr_with_input_id.rename(columns={'txId': 'input_id'})

# Merge to replace output_address with its txId
final_df = addr_with_input_id.merge(output_map, left_on='output_address', right_on='output_address', how='left')
final_df = final_df.drop(columns=['output_address', 'output_address'])
final_df = final_df.rename(columns={'txId': 'output_id'})

# Save the final result
final_df.to_csv('AddrAddr_with_txIds.csv', index=False)

# Load datasets
features = pd.read_csv(os.path.join(base_dir, 'txs_features.csv'), nrows=SAMPLE_ROWS)

classes = pd.read_csv(os.path.join(base_dir, 'txs_classes.csv'), nrows=SAMPLE_ROWS)

edges = pd.read_csv(os.path.join(base_dir, 'AddrAddr_with_txIds.csv'), nrows=SAMPLE_ROWS)

# Label mapping
label_map = {"1": 1, "2": 0, "unknown": -1}
classes["label"] = classes["class"].astype(str).map(label_map)

# Merge features and labels
data = pd.merge(features, classes, on="txId", how="left")

# Create mapping from txId to label
id_to_label = dict(zip(data["txId"], data["label"]))

# Build directed graph
G = nx.from_pandas_edgelist(edges, source="input_id", target="output_id", create_using=nx.DiGraph())

# Visual representation of the graph
sample_nodes = list(G.nodes)[:2000]  # Adjust this number if needed
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


# Compute degree features
deg_df = pd.DataFrame({
    "txId": list(G.nodes),
    "in_deg": [G.in_degree(n) for n in G.nodes],
    "out_deg": [G.out_degree(n) for n in G.nodes],
    "pagerank": pd.Series(nx.pagerank(G))
})

data = pd.merge(data, deg_df, on="txId", how="left")
data = data.fillna(0)  # fill pagerank etc.

# Filter labeled data
labeled_data = data[data["label"] != -1]

# Define features (excluding txId and label)
X = labeled_data.drop(columns=["txId", "label"])
X.columns = X.columns.astype(str)
y = labeled_data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("\nLabel distribution in training set:")
print(y_train.value_counts())

print("\nLabel distribution in test set:")
print(y_test.value_counts())

# Resample with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_res, y_train_res)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Licit", "Illicit"]))

# Feature Importance Plot
plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=15)
plt.title("Top 15 Most Important Features (XGBoost)")
plt.tight_layout()
plt.show()