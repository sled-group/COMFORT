from collections import defaultdict
import json
import os
from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser
from comfort_utils.helper import FOR_MAP, PERSPECTIVE_PROMPT_MAP
from comfort_utils.plot_helper import plot_spatial
from comfort_utils.symmetry_metric import (
    spatial_symmetry_metric,
    reverse_relation_symmetry_metric,
)
from comfort_utils.smoothness_metric import smoothness_lowpass_metric, normalize_data
from comfort_utils.accuracy_metric import accuracy_metric
from comfort_utils.language_ambiguity_metric import language_ambiguity_metric
from comfort_utils.perspective_taking_metric import perspective_taking_metric
from comfort_utils.convention_metric import convention_metric
from comfort_utils.std_metric import calculate_standard_deviation
from pandas import DataFrame

# import plotly.graph_objects as go
# import plotly.io as pio
# pio.kaleido.scope.mathjax = None # this line results AttributeError: 'NoneType' object has no attribute 'mathjax', commented.

# def calculate_default_variation_mean_abs_difference(
#     default_data: list[list[int, dict[str, float]]],
#     variation_data: list[list[int, dict[str, float]]],
# ):
#     default_yes_data = []
#     default_no_data = []
#     variation_yes_data = []
#     variation_no_data = []
#     for datapoint in default_data:
#         default_yes_data.append(datapoint[1]["Yes"])
#         default_no_data.append(datapoint[1]["No"])
#     for datapoint in variation_data:
#         variation_yes_data.append(datapoint[1]["Yes"])
#         variation_no_data.append(datapoint[1]["No"])
#     default_yes_data = normalize_data(np.array(default_yes_data))
#     variation_yes_data = normalize_data(np.array(variation_yes_data))
#     default_no_data = normalize_data(np.array(default_no_data))
#     variation_no_data = normalize_data(np.array(variation_no_data))
#     rmse_yes = np.sqrt(
#         np.mean((np.array(default_yes_data) - np.array(variation_yes_data)) ** 2)
#     )
#     rmse_no = np.sqrt(
#         np.mean((np.array(default_no_data) - np.array(variation_no_data)) ** 2)
#     )
#     return (rmse_yes + rmse_no) / 2

# deprecated
# def object_hall(all_results):
#     mean_yes_positive_list = []
#     mean_yes_negative_list = []
#     mean_no_positive_list = []
#     mean_no_negative_list = []
#     for configuration in all_results.keys():
#         results = all_results[configuration]["data"]
#         for variation_key in results:
#             variation = results[variation_key]
#             object1_hallucination_positive_yes_data = [
#                 data[1]["Yes"] for data in variation["object1_hallucination_positive"]
#             ]
#             object2_hallucination_positive_yes_data = [
#                 data[1]["Yes"] for data in variation["object2_hallucination_positive"]
#             ]
#             object_hallucination_negative_color_shape_yes_data = [
#                 data[1]["Yes"]
#                 for data in variation["object_hallucination_negative_color_shape"]
#             ]
#             object_hallucination_negative_color_yes_data = [
#                 data[1]["Yes"]
#                 for data in variation["object_hallucination_negative_color"]
#             ]
#             object_hallucination_negative_shape_yes_data = [
#                 data[1]["Yes"]
#                 for data in variation["object_hallucination_negative_shape"]
#             ]
#             object1_hallucination_positive_no_data = [
#                 data[1]["No"] for data in variation["object1_hallucination_positive"]
#             ]
#             object2_hallucination_positive_no_data = [
#                 data[1]["No"] for data in variation["object2_hallucination_positive"]
#             ]
#             object_hallucination_negative_color_shape_no_data = [
#                 data[1]["No"]
#                 for data in variation["object_hallucination_negative_color_shape"]
#             ]
#             object_hallucination_negative_color_no_data = [
#                 data[1]["No"]
#                 for data in variation["object_hallucination_negative_color"]
#             ]
#             object_hallucination_negative_shape_no_data = [
#                 data[1]["No"]
#                 for data in variation["object_hallucination_negative_shape"]
#             ]
#             mean_yes_positive = (
#                 np.mean(object1_hallucination_positive_yes_data)
#                 + np.mean(object2_hallucination_positive_yes_data)
#             ) / 2
#             mean_yes_negative = (
#                 np.mean(object_hallucination_negative_color_shape_yes_data)
#                 + np.mean(object_hallucination_negative_color_yes_data)
#                 + np.mean(object_hallucination_negative_shape_yes_data)
#             ) / 3
#             mean_no_positive = (
#                 np.mean(object1_hallucination_positive_no_data)
#                 + np.mean(object2_hallucination_positive_no_data)
#             ) / 2
#             mean_no_negative = (
#                 np.mean(object_hallucination_negative_color_shape_no_data)
#                 + np.mean(object_hallucination_negative_color_no_data)
#                 + np.mean(object_hallucination_negative_shape_no_data)
#             ) / 3
#             mean_yes_positive_list.append(mean_yes_positive)
#             mean_yes_negative_list.append(mean_yes_negative)
#             mean_no_positive_list.append(mean_no_positive)
#             mean_no_negative_list.append(mean_no_negative)
#     return (
#         np.mean(mean_yes_positive_list),
#         np.mean(mean_yes_negative_list),
#         np.mean(mean_no_positive_list),
#         np.mean(mean_no_negative_list),
#     )

def object_hall(all_results):
    yes_positive = 0
    yes_negative = 0
    no_positive = 0
    no_negative = 0
    for configuration in all_results.keys():
        results = all_results[configuration]["data"]
        for variation_key in results:
            variation = results[variation_key]
            object2_hallucination_positive_yes_data = [
                data[1]["Yes"] for data in variation["object2_hallucination_positive"]
            ]
            object_hallucination_negative_color_shape_yes_data = [
                data[1]["Yes"]
                for data in variation["object_hallucination_negative_color_shape"]
            ]
            object2_hallucination_positive_no_data = [
                data[1]["No"] for data in variation["object2_hallucination_positive"]
            ]
            object_hallucination_negative_color_shape_no_data = [
                data[1]["No"]
                for data in variation["object_hallucination_negative_color_shape"]
            ]
            for i in range(0, len(object2_hallucination_positive_yes_data)):
                if object2_hallucination_positive_yes_data[i] >= object2_hallucination_positive_no_data[i]:
                    yes_positive += 1
                else:
                    no_positive += 1
                if object_hallucination_negative_color_shape_yes_data[i] >= object_hallucination_negative_color_shape_no_data[i]:
                    yes_negative += 1
                else:
                    no_negative += 1
            
    return (
        yes_positive,
        yes_negative,
        no_positive,
        no_negative,
    )

def eval_metrics(all_results, save_path_root, perspective_prompt_type="camera", ref_rotation="left"):
    reverse_relation_symmetry_metric_avgs = []
    spatial_symmetry_metric_avgs = []
    smoothness_metric_avgs = []
    accuracy_metrics_normalized = []
    accuracy_metrics_clipped = []
    accuracy_metrics_thresholding = []
    accuracy_metrics_acc = []
    # default_variation_mean_abs_differences = defaultdict(list)
    stds = []

    for configuration in all_results.keys():
        results = all_results[configuration]["data"]
        variation_types = [entry for entry in results.keys()]
        # positive_default_data = results["default"]["positive"]
        # for variation_type in variation_types:
        #     default_variation_mean_abs_differences[variation_type].append(
        #         calculate_default_variation_mean_abs_difference(
        #             positive_default_data, results[variation_type]["positive"]
        #         )
        #     )
        stds.append(calculate_standard_deviation(results, variation_types))
        title = all_results[configuration]["positive_template"]
        plot_spatial(
            data_for_plot=[data["positive"] for data in results.values()],
            xlabel=all_results[configuration]["x_name"],
            title=title,
            config=all_results[configuration]["data"]["default"][
                    "config"
                ],
            FoR=FOR_MAP[ref_rotation][perspective_prompt_type][configuration],
            configuration=configuration,
            save_path=save_path_root,
            ref_rotation=ref_rotation
        )

    # calculate spatial_symmetry_metric, smoothness_lowpass_metric, accuracy_metric
    for configuration in all_results.keys():
        results = all_results[configuration]["data"]
        for variation in results.keys():
            data = results[variation]["positive"]

            # spatial_symmetry_metric
            spatial_symmetry_yes_metric = spatial_symmetry_metric(data, token="Yes")
            spatial_symmetry_metric_avg = spatial_symmetry_yes_metric
            spatial_symmetry_metric_avgs.append(spatial_symmetry_metric_avg)

            # smoothness_lowpass_metric
            smoothness_metric_avg = smoothness_lowpass_metric(data)
            smoothness_metric_avgs.append(smoothness_metric_avg)

            # accuracy_metric
            # assert perspective_prompt_type != "nop", "perspective_prompt_type cannot be nop"
            acc_normalized = accuracy_metric(
                data,
                config=all_results[configuration]["data"][variation][
                    "config"
                ],
                FoR=FOR_MAP[ref_rotation][perspective_prompt_type][configuration],
                mode="normalize"
            )
            accuracy_metrics_normalized.append(acc_normalized)
            acc_clipped = accuracy_metric(
                data,
                config=all_results[configuration]["data"][variation][
                    "config"
                ],
                FoR=FOR_MAP[ref_rotation][perspective_prompt_type][configuration],
                mode="clipped"
            )
            accuracy_metrics_clipped.append(acc_clipped)
            acc_thresholding = accuracy_metric(
                data,
                config=all_results[configuration]["data"][variation][
                    "config"
                ],
                FoR=FOR_MAP[ref_rotation][perspective_prompt_type][configuration],
                mode="thresholding"
            )
            accuracy_metrics_thresholding.append(acc_thresholding)
            acc_acc = accuracy_metric(
                data,
                config=all_results[configuration]["data"][variation][
                    "config"
                ],
                FoR=FOR_MAP[ref_rotation][perspective_prompt_type][configuration],
                mode="acc",
                normalize=False
            )
            accuracy_metrics_acc.append(acc_acc)

    # calculate reverse_relation_symmetry_metric
    negative_relations = {
        "totheleft": "totheright",
        "totheright": "totheleft",
        "behind": "infrontof",
        "infrontof": "behind",
    }
    for configuration in ["totheleft", "totheright", "behind", "infrontof"]:
        pos_results = all_results[configuration]["data"]
        neg_results = all_results[negative_relations[configuration]]["data"]
        for variation_type in pos_results.keys():
            pos_data = pos_results[variation_type]["positive"]
            neg_data = neg_results[variation_type]["positive"]
            reverse_relation_symmetry_yes_metric = reverse_relation_symmetry_metric(
                pos_data, neg_data, token="Yes", shift_angle=180
            )
            reverse_relation_symmetry_no_metric = reverse_relation_symmetry_metric(
                pos_data, neg_data, token="No", shift_angle=180
            )
            reverse_relation_symmetry_metric_avg = (
                reverse_relation_symmetry_yes_metric
                + reverse_relation_symmetry_no_metric
            ) / 2
            reverse_relation_symmetry_metric_avgs.append(
                reverse_relation_symmetry_metric_avg
            )
    # for variation_type in default_variation_mean_abs_differences.keys():
    #     default_variation_mean_abs_differences[variation_type] = np.mean(
    #         default_variation_mean_abs_differences[variation_type]
    #     )
    stds = np.array(stds)
    # print("stds:", stds.shape)
    # print(np.array(accuracy_metrics).shape)
    return (
        np.mean(reverse_relation_symmetry_metric_avgs),
        np.mean(spatial_symmetry_metric_avgs),
        np.mean(smoothness_metric_avgs),
        np.mean(accuracy_metrics_normalized),
        np.mean(accuracy_metrics_clipped),
        np.mean(accuracy_metrics_thresholding),
        np.mean(accuracy_metrics_acc),
        np.mean(stds)
    )


def eval_hard_metrics(all_results_nop, all_results_camera, all_results_reference, all_results_addressee, mode, ref_rotation):
    hard_metrics_by_configurations = {}
    for configuration in all_results_nop.keys():
        results_nop = all_results_nop[configuration]["data"]
        perspective_taking_metric_values = []
        # print("eval_hard_metrics - variations:", results_nop.keys())
        # print("eval_hard_metrics - # variations:", len(results_nop.keys()))
        for variation in results_nop.keys():
            perspective_taking_metric_values.append(
                perspective_taking_metric(
                    all_results_camera, all_results_reference, all_results_addressee, configuration, variation, mode, ref_rotation
                )
            )
        language_ambiguity_metric_value = language_ambiguity_metric(
            all_results_nop, configuration, mode, ref_rotation
        )
        hard_metrics_by_configurations[configuration] = [
            perspective_taking_metric_values,
            language_ambiguity_metric_value
        ]
    return hard_metrics_by_configurations

def eval_conventions(all_results_camera, mode, ref_rotation):
    metrics_by_configurations = {}
    for configuration in all_results_camera.keys():
        results_camera = all_results_camera[configuration]["data"]
        convention_metric_values = []
        # print("eval_hard_metrics - variations:", results_nop.keys())
        # print("eval_hard_metrics - # variations:", len(results_nop.keys()))
        for variation in results_camera.keys():
            convention_metric_values.append(
                convention_metric(
                    all_results_camera, configuration, variation, mode, ref_rotation
                )
            )
        metrics_by_configurations[configuration] = convention_metric_values
    return metrics_by_configurations


def spatial_eval(results_path, hard_metrics=False, cosmode="softcos", convention_eval=False, ref_rotation="left"):
    plots_root = "plots/" + "/".join(results_path.split("/")[1:-1])
    if convention_eval:
        camera_path = os.path.join(results_path, "camera3")
        model_names = []
        metrics_list = []
        for file in os.listdir(camera_path):
            if file.endswith(".json"):
                model_name = file.split(".json")[0].replace("_", "-")
                model_names.append(model_name)
                all_results_camera_path = os.path.join(camera_path, file)
                if (
                    os.path.exists(all_results_camera_path)
                ):
                    with open(all_results_camera_path, "r") as file:
                        all_results_camera = json.load(file)
                    # print(
                    #     f"{all_results_nop_path}, {all_results_camera_path}, {all_results_reference_path}, {all_results_addressee_path} loaded successfully."
                    # )
                else:
                    raise Exception(
                        "Please provide a results_path (excluding perspective prompt type) for calculating hard metric. One or more perspective files do not exist."
                    )
                all_results_camera.pop("dataset_type")
                if "dataset_type_full" in all_results_camera:
                    all_results_camera.pop("dataset_type_full")
                all_results_camera.pop("model")
                metrics_list.append(eval_conventions(
                        all_results_camera, cosmode, ref_rotation
                    )
                )
        return model_names, metrics_list
    else:
        if hard_metrics:
            nop_path = os.path.join(results_path, "nop")
            camera_path = os.path.join(results_path, "camera3")
            reference_path = os.path.join(results_path, "reference3")
            addressee_path = os.path.join(results_path, "addressee3")
            model_names = []
            metrics_list = []
            for file in os.listdir(nop_path):
                if file.endswith(".json"):
                    model_name = file.split(".json")[0].replace("_", "-")
                    model_names.append(model_name)
                    all_results_nop_path = os.path.join(nop_path, file)
                    all_results_camera_path = os.path.join(camera_path, file)
                    all_results_reference_path = os.path.join(reference_path, file)
                    all_results_addressee_path = os.path.join(addressee_path, file)
                    if (
                        os.path.exists(all_results_nop_path)
                        and os.path.exists(all_results_camera_path)
                        and os.path.exists(all_results_reference_path)
                        and os.path.exists(all_results_addressee_path)
                    ):
                        with open(all_results_nop_path, "r") as file:
                            all_results_nop = json.load(file)
                        with open(all_results_camera_path, "r") as file:
                            all_results_camera = json.load(file)
                        with open(all_results_reference_path, "r") as file:
                            all_results_reference = json.load(file)
                        with open(all_results_addressee_path, "r") as file:
                            all_results_addressee = json.load(file)
                        # print(
                        #     f"{all_results_nop_path}, {all_results_camera_path}, {all_results_reference_path}, {all_results_addressee_path} loaded successfully."
                        # )
                    else:
                        raise Exception(
                            "Please provide a results_path (excluding perspective prompt type) for calculating hard metric. One or more perspective files do not exist."
                        )
                    all_results_nop.pop("dataset_type")
                    if "dataset_type_full" in all_results_nop:
                        all_results_nop.pop("dataset_type_full")
                    all_results_camera.pop("dataset_type")
                    if "dataset_type_full" in all_results_camera:
                        all_results_camera.pop("dataset_type_full")
                    all_results_reference.pop("dataset_type")
                    if "dataset_type_full" in all_results_reference:
                        all_results_reference.pop("dataset_type_full")
                    all_results_addressee.pop("dataset_type")
                    if "dataset_type_full" in all_results_addressee:
                        all_results_addressee.pop("dataset_type_full")
                    all_results_nop.pop("model")
                    all_results_camera.pop("model")
                    all_results_reference.pop("model")
                    all_results_addressee.pop("model")
                    metrics_list.append(
                        eval_hard_metrics(
                            all_results_nop, all_results_camera, all_results_reference, all_results_addressee, cosmode, ref_rotation
                        )
                    )
            return model_names, metrics_list
        else:
            # if results_path is not None and results_path.endswith(".json"):
            if os.path.exists(results_path):
                # Load the JSON data into the result variable
                with open(results_path, "r") as file:
                    all_results = json.load(file)
                # print(f"{results_path} loaded successfully.")
            else:
                print(f"{results_path} does not exist. Nothing to plot.")
                exit()
            perspective = results_path.split("/")[-2]
            perspective_prompt_type = PERSPECTIVE_PROMPT_MAP[perspective]
            model_name = all_results.pop("model").split("/")[-1]
            all_results.pop("dataset_type")
            if "dataset_type_full" in all_results:
                all_results.pop("dataset_type_full")
            save_path_root = os.path.join(plots_root, model_name)
            if not os.path.exists(save_path_root):
                os.makedirs(save_path_root)

            (
                mean_yes_positive,
                mean_yes_negative,
                mean_no_positive,
                mean_no_negative,
            ) = object_hall(all_results)
            # print("yes_positive:", mean_yes_positive)
            # print("yes_negative:", mean_yes_negative)
            # print("no_positive:", mean_no_positive)
            # print("no_negative:", mean_no_negative)
            # # debug
            # print("reverse_relation_symmetry_metric_avgs:", reverse_relation_symmetry_metric_avgs)
            # print("spatial_symmetry_metric_avgs:", spatial_symmetry_metric_avgs)
            # print("smoothness_metric_avgs:", smoothness_metric_avgs)
            (
                reverse_relation_symmetry_metric_avgs,
                spatial_symmetry_metric_avgs,
                smoothness_metric_avgs,
                accuracy_metrics_normalized,
                accuracy_metrics_clipped,
                accuracy_metrics_thresholding,
                accuracy_metrics_acc,
                std,
            ) = eval_metrics(all_results, save_path_root, perspective_prompt_type, ref_rotation)
            # p_std_all = []
            # for variation_type in default_variation_mean_abs_differences.keys():
            #     p_std = default_variation_mean_abs_differences[variation_type]
            #     # print(
            #     #     f"mean {variation_type} variation rmse differences:",
            #     #     p_std,
            #     # )
            #     p_std_all.append(p_std)
            # print(f"mean variation rmse differences:", np.mean(p_std_all))
            # print(
            #     "mean reverse relation symmetry metric avg:",
            #     reverse_relation_symmetry_metric_avgs,
            # )
            # print("mean spatial symmetry metric avg:", spatial_symmetry_metric_avgs)
            # print("mean smoothness metric avg:", smoothness_metric_avgs)
            # print("mean accuracy:", accuracy_metrics)
            return {
                "mean_yes_positive": mean_yes_positive,
                "mean_no_negative": mean_no_negative,
                "mean_yes_negative": mean_yes_negative,
                "mean_no_positive": mean_no_positive,
                "std": std,
                "reverse_relation_symmetry_metric_avgs": reverse_relation_symmetry_metric_avgs,
                "spatial_symmetry_metric_avgs": spatial_symmetry_metric_avgs,
                "smoothness_metric_avgs": smoothness_metric_avgs,
                "accuracy_metrics_normalized": accuracy_metrics_normalized,
                "accuracy_metrics_clipped": accuracy_metrics_clipped,
                "accuracy_metrics_thresholding": accuracy_metrics_thresholding,
                "accuracy_metrics_acc": accuracy_metrics_acc
            }
            # else:
            #     metrics = {
            #         "model": [],
            #         "mean_yes_positive": [],
            #         "mean_yes_negative": [],
            #         "mean_no_positive": [],
            #         "mean_no_negative": [],
            #         "mean_reverse_relation_symmetry_metric_avg": [],
            #         "mean_spatial_symmetry_metric_avg": [],
            #         "mean_smoothness_metric_avg": [],
            #         "mean_accuracy": [],
            #         "mean_variation_rmse_diff": [],
            #     }
            #     for results in sorted(os.listdir(results_root)):
            #         if results.endswith(".json"):
            #             results_path = os.path.join(results_root, results)
            #         else:
            #             continue
            #         if os.path.exists(results_path):
            #             # Load the JSON data into the result variable
            #             with open(results_path, "r") as file:
            #                 all_results = json.load(file)
            #             print(f"{results_path} loaded successfully.")
            #         else:
            #             print(f"{results_path} does not exist. Continuing.")
            #             continue
            #         model_name = all_results.pop("model").split("/")[-1]
            #         save_path_root = os.path.join(plots_root, model_name)
            #         if not os.path.exists(save_path_root):
            #             os.makedirs(save_path_root)
            #         (
            #             mean_yes_positive,
            #             mean_yes_negative,
            #             mean_no_positive,
            #             mean_no_negative,
            #         ) = object_hall(all_results)
            #         print("yes_positive:", mean_yes_positive)
            #         print("yes_negative:", mean_yes_negative)
            #         print("no_positive:", mean_no_positive)
            #         print("no_negative:", mean_no_negative)
            #         # # debug
            #         # print("reverse_relation_symmetry_metric_avgs:", reverse_relation_symmetry_metric_avgs)
            #         # print("spatial_symmetry_metric_avgs:", spatial_symmetry_metric_avgs)
            #         # print("smoothness_metric_avgs:", smoothness_metric_avgs)
            #         (
            #             reverse_relation_symmetry_metric_avgs,
            #             spatial_symmetry_metric_avgs,
            #             smoothness_metric_avgs,
            #             accuracy_metrics,
            #             default_variation_mean_abs_differences,
            #         ) = eval_metrics(all_results, save_path_root, dataset_type)
            #         p_std_all = []
            #         for variation_type in default_variation_mean_abs_differences.keys():
            #             p_std = default_variation_mean_abs_differences[variation_type]
            #             print(
            #                 f"mean {variation_type} variation rmse differences:",
            #                 p_std,
            #             )
            #             p_std_all.append(p_std)
            #         print(f"mean variation rmse differences:", np.mean(p_std_all))
            #         print(
            #             "mean reverse relation symmetry metric avg:",
            #             reverse_relation_symmetry_metric_avgs,
            #         )
            #         print("mean spatial symmetry metric avg:", spatial_symmetry_metric_avgs)
            #         print("mean smoothness metric avg:", smoothness_metric_avgs)
            #         print("mean accuracy:", accuracy_metrics)
            #         metrics["model"].append(model_name)
            #         metrics["mean_yes_positive"].append(mean_yes_positive)
            #         metrics["mean_yes_negative"].append(mean_yes_negative)
            #         metrics["mean_no_positive"].append(mean_no_positive)
            #         metrics["mean_no_negative"].append(mean_no_negative)
            #         metrics["mean_reverse_relation_symmetry_metric_avg"].append(
            #             reverse_relation_symmetry_metric_avgs
            #         )
            #         metrics["mean_spatial_symmetry_metric_avg"].append(
            #             spatial_symmetry_metric_avgs
            #         )
            #         metrics["mean_smoothness_metric_avg"].append(smoothness_metric_avgs)
            #         metrics["mean_accuracy"].append(accuracy_metrics)
            #         metrics["mean_variation_rmse_diff"].append(np.mean(p_std_all))
            #     df = DataFrame(metrics)
            #     df.to_csv(os.path.join(args.results_root, "metrics.csv"), index=False)
            #     df_melted = df.melt(id_vars='model', var_name='metric', value_name='value')
            #     fig = go.Figure()
            #     models = df['model'].unique()
            #     for model in models:
            #         filtered_df = df_melted[df_melted['model'] == model]
            #         fig.add_trace(go.Bar(
            #             x=filtered_df['metric'],
            #             y=filtered_df['value'],
            #             name=model,
            #         ))
            #     fig.update_layout(
            #         barmode='group',
            #         title='Metrics all',
            #         xaxis_title='Metric',
            #         yaxis_title='Value',
            #         legend_title='Model'
            #     )
            #     fig.write_image(os.path.join(plots_root, "metrics.pdf"))
