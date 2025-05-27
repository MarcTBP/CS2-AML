import numpy as np

# Load the correct .npz file
npz_path = r"data\np\addr0_tx_mean\addr_feature_np_list.npz"
data = np.load(npz_path, allow_pickle=True)

print("Keys in the .npz file:", data.files)

# Use the key for the test set features
key = "addr_features_test_list"

if key in data:
    features_list = data[key]
    print(f"Total time steps: {len(features_list)}")
    for i in range(min(3, len(features_list))):
        step_data = features_list[i]
        print(f"\nTime Step {i + 1}")
        print("Shape:", step_data.shape)
        print("First row (truncated):", step_data[0][:10])
        print("Last 17 features (targeted for ablation):", step_data[0][-17:])
else:
    print(f"Key '{key}' not found in {npz_path}")
