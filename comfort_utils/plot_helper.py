import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from .symmetry_metric import normalize_data
from .helper import FOR_MAP

def _phase_shift_notrot2mixed(x: list[float], save_path):
    assert len(x) == 37, f"Expected len(x) == 37, got {len(x)}"
    # if np.abs(x[0] - x[-1]) < 0.01:
        # f"Expected |x[0]-x[-1]| < 0.01, got {x[0]}, {x[-1]}"
        # print("save path:", save_path)
    result = []
    _x = x[:-1]
    middle = (len(_x)) // 2 # 18
    for i in range(len(_x)):
        result.append(_x[(i + middle) % len(_x)])
    result.append(result[0])
    return result

title_matching = {
    "Is the [a] to the left of the [b]?": "Is [A] to the left of [B]?",
    "Is the [a] to the right of the [b]?": "Is [A] to the right of [B]?",
    "Is the [a] in front of the [b]?": "Is [A] in front of [B]?",
    "Is the [a] behind the [b]?": "Is [A] behind [B]?"
}

def plot_spatial(
    data_for_plot, xlabel, title, FoR, config, configuration, save_path=None, show=False, normalize=False, ref_rotation="left"
):
    for_shift = {
        "camera": 0,
        "addressee": 90,
        "rotated_camera": 180,
        "rotated_addressee": 270,
        "object_facing_right": 90,
        "object_facing_left": 270,
    }
    assert FoR in for_shift, "FoR must be one of camera, object, addressee, rotated_camera"
    assert config["relation"] in [
        "infrontof",
        "behind",
        "totheleft",
        "totheright",
    ], "relation must be one of infrontof, behind, totheleft, totheright"

    normalized_data_for_plot = [normalize_data(data) for data in data_for_plot]
    attribute_data = {}
    normalized_attribute_data = {}
    gts = {} # gt means ground-truth
    gts["soft"] = {"gt": {}, "gt_camera": {}, "gt_addressee": {}, "gt_reference": {}}
    gts["hard"] = {"gt": {}, "gt_camera": {}, "gt_addressee": {}, "gt_reference": {}}
    gts["clipped"] = {"gt": {}, "gt_camera": {}, "gt_addressee": {}, "gt_reference": {}}
    x_values = [
        round(point[0][0], 3) if isinstance(point[0], list) else round(point[0], 3)
        for point in data_for_plot[0]
    ]
    for experiment in data_for_plot:
        for i, point in enumerate(experiment):
            # point is a tuple of the form (angle, {token1:likelihood1, token2:likelihood2})
            x_value = x_values[i]
            for token, likelihood in point[1].items():
                if token != "No":
                    if token not in attribute_data:
                        attribute_data[token] = {}
                        for cos_mode in gts.keys():
                            for gt_type in gts[cos_mode].keys():
                                gts[cos_mode][gt_type][token] = {}
                    if x_value not in attribute_data[token]:
                        attribute_data[token][x_value] = []
                        for cos_mode in gts.keys():
                            for gt_type in gts[cos_mode].keys():
                                gts[cos_mode][gt_type][token][x_value] = []
                    attribute_data[token][x_value].append(likelihood)
                    shift = for_shift[FoR]
                    shift_camera = for_shift[FOR_MAP[ref_rotation]["camera"][configuration]]
                    shift_reference = for_shift[FOR_MAP[ref_rotation]["object"][configuration]]
                    shift_addressee = for_shift[FOR_MAP[ref_rotation]["addressee"][configuration]]
                    cosine = np.cos((x_value + shift) / 180 * np.pi)
                    cosine_camera = np.cos((x_value + shift_camera) / 180 * np.pi)
                    cosine_addressee = np.cos((x_value + shift_addressee) / 180 * np.pi)
                    cosine_reference = np.cos((x_value + shift_reference) / 180 * np.pi)
                    if np.abs(cosine) < 1e-10:
                        cosine = 0
                    if np.abs(cosine_camera) < 1e-10:
                        cosine_camera = 0
                    if np.abs(cosine_addressee) < 1e-10:
                        cosine_addressee = 0
                    if np.abs(cosine_reference) < 1e-10:
                        cosine_reference = 0
                    gt_curves = {"gt": cosine,
                                "gt_camera": cosine_camera,
                                "gt_addressee": cosine_addressee,
                                "gt_reference": cosine_reference}
                    for cos_mode in gts.keys():
                        if cos_mode == "soft":
                            for gt_type in gts[cos_mode].keys():
                                gts[cos_mode][gt_type][token][x_value].append((gt_curves[gt_type] + 1) / 2)
                        elif cos_mode == "hard":
                            for gt_type in gts[cos_mode].keys():
                                gt_temp = np.zeros_like(gt_curves[gt_type])
                                gt_temp[gt_curves[gt_type] > 0] = 1
                                gts[cos_mode][gt_type][token][x_value].append(gt_temp)
                        elif cos_mode == "clipped":
                            for gt_type in gts[cos_mode].keys():
                                gts[cos_mode][gt_type][token][x_value].append(np.clip(gt_curves[gt_type], 0, 1))
    mean_errs = {}
    for cos_mode in gts.keys():
        for experiment in normalized_data_for_plot:
            for i, point in enumerate(experiment):
                # point is a tuple of the form (angle, {token1:likelihood1, token2:likelihood2})
                x_value = x_values[i]
                for token, likelihood in point[1].items():
                    if token != "No":
                        if token not in normalized_attribute_data:
                            normalized_attribute_data[token] = {}
                        if x_value not in normalized_attribute_data[token]:
                            normalized_attribute_data[token][x_value] = []
                        normalized_attribute_data[token][x_value].append(likelihood)
        num_data_points_counter = 0
        error = 0
        for experiment in normalized_data_for_plot[:1]:
            for i, point in enumerate(experiment):
                # point is a tuple of the form (angle, {token1:likelihood1, token2:likelihood2})
                x_value = x_values[i]
                for token, likelihood in point[1].items():
                    if token != "No":
                        shift = for_shift[FoR]
                        gt = np.cos((x_value + shift) / 180 * np.pi)
                        if np.abs(gt) < 1e-10:
                            gt = 0
                        if cos_mode == "soft":
                            gt_new = (gt + 1) / 2 
                        elif cos_mode == "hard":
                            if gt > 0:
                                gt_new = 1
                            else:
                                gt_new = 0
                        elif cos_mode == "clipped":
                            gt_new = np.clip(gt, 0, 1)
                        error += (gt_new - np.mean(normalized_attribute_data[token][x_value])) ** 2
                        num_data_points_counter += 1
                        # print("x_value:", x_value)
                        # print("mean likelihood:", np.mean(normalized_attribute_data[token][x_value]))
                        # print("gt:", gt_new)
                        # print("cumulative error:", error)
            mean_errs[cos_mode] = np.sqrt(error/num_data_points_counter)

    ###### Plotting Parameters ######
    only_legend = False
    show_err = True
    err_type = "soft" # "soft" or "hard"
    data_types_plot_or_not = {
        "data": True,
        "normalized_data": True,
        "soft_gt": False,
        "soft_gt_camera": True,
        "soft_gt_addressee": True,
        "soft_gt_reference": True,
        "hard_gt": False,
        "hard_gt_camera": False,
        "hard_gt_addressee": False,
        "hard_gt_reference": False,
        "clipped_gt": False,
        "clipped_gt_camera": False,
        "clipped_gt_addressee": False,
        "clipped_gt_reference": False 
    }
    ###### Plotting Parameters ######

    plt.rcParams['font.family'] = 'Liberation Serif'
    plt.rcParams["legend.loc"] = 'lower right'
    plt.figure(figsize=(10, 15))
    plt.tight_layout(pad=1.0)
    sns.set_context("notebook", font_scale=2)
    _, ax = plt.subplots()

    all_data = {"data": attribute_data, "normalized_data": normalized_attribute_data}
    data_type_color_mapping = {
        "data": sns.color_palette()[7], # gray
        "normalized_data": "black",
        "soft_gt": "red",
        "soft_gt_camera": "red",
        "soft_gt_addressee": sns.color_palette("bright")[1], # orange
        "soft_gt_reference": "blue",
        "hard_gt": "red",
        "hard_gt_camera": sns.color_palette("bright")[4],
        "hard_gt_addressee": sns.color_palette("bright")[1], # orange
        "hard_gt_reference": "blue",
        "clipped_gt": "red",
        "clipped_gt_camera": "red",
        "clipped_gt_addressee": sns.color_palette("bright")[1], # orange
        "clipped_gt_reference": "blue"
    }

    for cos_mode in gts.keys():
        for gt_type in gts[cos_mode].keys():
            all_data[cos_mode + '_' + gt_type] = gts[cos_mode][gt_type]

    for data_type in all_data.keys():
        this_attribute_data = all_data[data_type]
        for attribute, values in this_attribute_data.items():
            x = sorted(values.keys())
            # print("y:", values)
            y_means = [np.mean(values[x_val]) for x_val in x]
            y_stds = [np.std(values[x_val]) for x_val in x]
            x_float = np.asarray(x, dtype=float)
            if "in front of" in title or "behind" in title:
                y_means = _phase_shift_notrot2mixed(y_means, save_path)
                y_stds = _phase_shift_notrot2mixed(y_stds, save_path)
            if data_type == "data":
                if data_types_plot_or_not[data_type]:
                    if only_legend:
                        sns.lineplot(x=x_float, y=[0.0 for _ in x_float], label="VLM Original Prediction", color=data_type_color_mapping[data_type], linewidth=3)
                    else:
                        sns.lineplot(x=x_float, y=y_means, label="VLM Original Prediction", color=data_type_color_mapping[data_type], linewidth=3)
                        plt.fill_between(x_float, np.subtract(y_means, y_stds), np.add(y_means, y_stds), color=data_type_color_mapping[data_type], alpha=0.2)
                else:
                    pass
            elif data_type == "normalized_data":
                if data_types_plot_or_not[data_type]:
                    if only_legend:
                        sns.lineplot(x=x_float, y=[0.0 for _ in x_float], label="VLM Normalized Prediction", color=data_type_color_mapping[data_type], linewidth=3)
                    else:
                        sns.lineplot(x=x_float, y=y_means, label="VLM Normalized Prediction", color=data_type_color_mapping[data_type], linewidth=3)
                        plt.fill_between(x_float, np.subtract(y_means, y_stds), np.add(y_means, y_stds), color=data_type_color_mapping[data_type], alpha=0.2)
                else:
                    pass
            elif len(data_type.split('_')) == 2 and data_type.split('_')[1] == "gt":
                if data_types_plot_or_not[data_type]:
                    if only_legend:
                        sns.lineplot(x=x_float, y=[0.0 for _ in x_float], label="Ground Truth", color=data_type_color_mapping[data_type], linestyle='--')
                    else:
                        sns.lineplot(x=x_float, y=y_means, label="Ground Truth", color=data_type_color_mapping[data_type], linestyle='--')
                else:
                    pass
            elif len(data_type.split('_')) == 3 and data_type.split('_')[2] == "camera":
                if data_types_plot_or_not[data_type]:
                    if only_legend:
                        sns.lineplot(x=x_float, y=[0.0 for _ in x_float], label=f"Camera ({data_type.split('_')[0]})", color=data_type_color_mapping[data_type], linestyle='--')
                    else:
                        sns.lineplot(x=x_float, y=y_means, label=f"Camera ({data_type.split('_')[0]})", color=data_type_color_mapping[data_type], linestyle='--')
                else:
                    pass
            elif len(data_type.split('_')) == 3 and data_type.split('_')[2] == "addressee":
                if data_types_plot_or_not[data_type]:
                    if only_legend:
                        sns.lineplot(x=x_float, y=[0.0 for _ in x_float], label=f"Addressee ({data_type.split('_')[0]})", color=data_type_color_mapping[data_type], linestyle='--')
                    else:
                        sns.lineplot(x=x_float, y=y_means, label=f"Addressee ({data_type.split('_')[0]})", color=data_type_color_mapping[data_type], linestyle='--')
                else:
                    pass
            elif len(data_type.split('_')) == 3 and data_type.split('_')[2] == "reference":
                if data_types_plot_or_not[data_type]:
                    if only_legend:
                        sns.lineplot(x=x_float, y=[0.0 for _ in x_float], label=f"Reference ({data_type.split('_')[0]})", color=data_type_color_mapping[data_type], linestyle='--')
                    else:
                        sns.lineplot(x=x_float, y=y_means, label=f"Reference ({data_type.split('_')[0]})", color=data_type_color_mapping[data_type], linestyle='--')
                else:
                    pass

    ax.set_xlim([-180, 180])
    ax.set_ylim([0, 1])
    plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel(r"angle")
    ax.set_ylabel(r"probability") # ax.set_ylabel(r"normalized probability")
    # ax.set_title(r'\textbf{' + title_matching[title[29:].capitalize()] + "}")
    if show_err:
        ax.set_title(r"$\varepsilon_{gt}$: " + f"{mean_errs[err_type]*100:.1f}")
    # ax.legend()
    # plt.tight_layout()
    if not only_legend:
        ax.get_legend().remove()
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if "left" in title:
            shortened_title = "left"
        elif "right" in title:
            shortened_title = "right"
        elif "in front of" in title:
            shortened_title = "infrontof"
        elif "behind" in title:
            shortened_title = "behind"
        plt.savefig(os.path.join(save_path, f"{shortened_title}.pdf"), bbox_inches="tight", transparent=True) # _err_{mean_errs[err_type]}
        # plt.show()
        plt.close()
    if show:
        plt.show()
