import os
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from numpy.linalg import norm
from scipy.signal import butter, filtfilt


def low_pass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y_filtered = filtfilt(b, a, data)
    return y_filtered


def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


# def simple_plot(x_data, y_data, fitted_y):
#     plt.figure(figsize=(10, 5))
#     plt.scatter(x_data, y_data, color="blue", label="Data Points")
#     plt.plot(x_data, fitted_y, color="red", label="Fitted Curve")
#     plt.title("Curve Fitting")
#     plt.legend()
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.show()


# def fit_function(x, A, B, C):
#     return A * np.sin(B * x) + C


# def calculate_curve_fit_error(data):
#     x_data = []
#     yes_data = []
#     no_data = []
#     for sub_data in data:
#         x_data.append(sub_data[0])
#         yes_data.append(sub_data[2])
#         no_data.append(sub_data[4])
#     x_data = np.array(x_data)
#     x_data = x_data.astype(float)
#     yes_data = np.array(yes_data)
#     no_data = normalize_data(np.array(no_data))
#     popt, pcov = curve_fit(fit_function, x_data, yes_data, p0=[0.01, 0.01, 0.01])
#     print(popt)
#     fitted_y = fit_function(x_data, *popt)
#     l2_diff = np.linalg.norm(yes_data - fitted_y)
#     print(l2_diff)
#     simple_plot(x_data, yes_data, fitted_y)
#     return l2_diff


# def calculate_curve_fit_error_2(data):
#     x_data = []
#     yes_data = []
#     no_data = []
#     for sub_data in data:
#         x_data.append(sub_data[0])
#         yes_data.append(sub_data[2])
#         no_data.append(sub_data[4])
#     x_data = np.array(
#         x_data,
#     )
#     x_data = x_data.astype(float)
#     yes_data = normalize_data(np.array(yes_data))
#     no_data = normalize_data(np.array(no_data))
#     s_alpha_hat = fit_sin_model(x_data, yes_data, frequencies)
#     y_pred = sin_model_predictions(x_data, s_alpha_hat, frequencies)
#     simple_plot(x_data, yes_data, y_pred)
#     l2_distance = norm(yes_data - y_pred)
#     return l2_distance


def smoothness_lowpass_metric(data):
    assert len(data) == 37, f"Data length should be 37, got: {len(data)}"
    if len(data) == 37:
        data = data[:-1]
    order = 5
    cutoff = 100  # desired cutoff Hz
    x_data = []
    yes_data = []
    # no_data = []
    for sub_data in data:
        if type(sub_data[0]) == float or type(sub_data[0]) == int:
            x_data.append(sub_data[0])
        else:
            x_data.append(sub_data[0][0])
        yes_data.append(sub_data[1]["Yes"])
        # no_data.append(sub_data[1]["No"])
    x_data = np.array(
        x_data,
    )
    x_data = x_data.astype(float)
    yes_data = normalize_data(np.array(yes_data))
    # no_data = normalize_data(np.array(no_data))
    filtered_yes_data = low_pass_filter(yes_data, cutoff, fs=1000, order=order)
    # filtered_no_data = low_pass_filter(no_data, cutoff, fs=1000, order=order)
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_data, yes_data, label="Original Signal")
    # plt.plot(x_data, filtered_data, label="Filtered Signal", linewidth=2)
    # plt.title("Signal Before and After Low-Pass Filtering")
    # plt.xlabel("x")
    # plt.ylabel("prob")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    rmse_yes = np.sqrt(np.mean((np.array(yes_data) - np.array(filtered_yes_data)) ** 2))
    # rmse_no = np.sqrt(np.mean((np.array(no_data) - np.array(filtered_no_data)) ** 2))
    # mse_yes = np.mean((np.array(yes_data) - np.array(filtered_yes_data))**2)
    # mse_no = np.mean((np.array(no_data) - np.array(filtered_no_data))**2)
    # return (rmse_yes + rmse_no) / 2
    return rmse_yes


def fit_sin_model(x, y, frequencies):
    S = np.vstack([np.sin(f * x) for f in frequencies] + [np.ones(len(x))]).T
    s_alpha_hat, _, _, _ = lstsq(S, y)
    return s_alpha_hat


def sin_model_predictions(t, coefficients, frequencies):
    return (
        np.sum(
            [coeff * np.sin(f * t) for coeff, f in zip(coefficients[:-1], frequencies)],
            axis=0,
        )
        + coefficients[-1]
    )


def fit_cos_model(x, y, frequencies):
    S = np.vstack([np.cos(f * x) for f in frequencies] + [np.ones(len(x))]).T
    s_alpha_hat, _, _, _ = lstsq(S, y)
    return s_alpha_hat


def cos_model_predictions(t, coefficients, frequencies):
    return (
        np.sum(
            [coeff * np.cos(f * t) for coeff, f in zip(coefficients[:-1], frequencies)],
            axis=0,
        )
        + coefficients[-1]
    )


if __name__ == "__main__":
    results_path = "spatial_eval_results.json"
    if os.path.exists(results_path):
        # Load the JSON data into the result variable
        with open(results_path, "r") as file:
            results = json.load(file)
        print("spatial_eval_results.json loaded successfully.")
    else:
        print("spatial_eval_results.json does not exist. Nothing to plot.")
        exit()
    model_name = results.pop("model").split("/")[-1].split("-")[0]
    save_path_root = f"plots/spatial_plots/{model_name}"
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    frequencies = [5]
    print(results)
    for configuration in results.keys():
        positive_variations_data = []
        negative_variations_data = []
        variation_types = [
            entry
            for entry in results[configuration].keys()
            if entry != "x_name"
            and entry != "positive_template"
            and entry != "negative_template"
        ]
        color_variation_index = variation_types.index("color")
        shape_variation_index = variation_types.index("shape")
        default_variation_index = variation_types.index("default")
        for variation_type in variation_types:
            print(configuration)
            print(variation_type)
            print(
                smoothness_lowpass_metric(
                    results[configuration][variation_type]["positive"]
                )
            )

            if "negative" in results[configuration][variation_type]:
                # results[configuration][variation_type]["negative"]
                pass
