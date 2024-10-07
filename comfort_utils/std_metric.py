import numpy as np
from .smoothness_metric import normalize_data

def calculate_mean(datasets: list[np.ndarray]) -> np.ndarray:
    combined_data = np.stack(datasets, axis=0)
    mean_data = np.mean(combined_data, axis=0)
    return mean_data

def calculate_standard_deviation_helper(datasets: list[np.ndarray], mean_data: np.ndarray) -> np.ndarray:
    combined_data = np.stack(datasets, axis=0)
    # print("combined_data shape:", combined_data.shape)
    # print("mean data shape:", mean_data.shape)
    differences = combined_data - mean_data
    # print("combined_data:", combined_data)
    # print("mean_data:", mean_data)
    # print("differences:", differences)
    # print("differences shape:", differences.shape)
    squared_differences = np.square(differences)
    mean_squared_diff = np.mean(squared_differences)
    std_dev = np.sqrt(mean_squared_diff)
    return std_dev

def calculate_standard_deviation(results, variation_types):
    all_yes_data = []

    for variation_type in variation_types:
        variation_yes_data = [datapoint[1]["Yes"] for datapoint in results[variation_type]["positive"]]
        variation_yes_data = normalize_data(np.array(variation_yes_data))
        assert len(variation_yes_data) == 37, f"Data length should be 37, got: {len(variation_yes_data)}"
        if len(variation_yes_data) == 37:
            variation_yes_data = variation_yes_data[:-1]
        all_yes_data.append(variation_yes_data)
    # print("all yes data shape:", np.array(all_yes_data).shape)

    mean_yes_data = calculate_mean(all_yes_data)
    # print("mean_yes_data shape:", np.array(mean_yes_data).shape)

    std_dev_yes = calculate_standard_deviation_helper(all_yes_data, mean_yes_data)

    return std_dev_yes