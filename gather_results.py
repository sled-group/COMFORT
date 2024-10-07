from spatial_eval import spatial_eval
from argparse import ArgumentParser
from comfort_utils.convert_general_metrics_results import convert_general_metrics_results
from comfort_utils.convert_perspective_taking_results import convert_perspective_taking_results
import os
import numpy as np
import pandas as pd

model_name_mapping = {"instructblip-vicuna-7b": "InstructBLIP-7B",
                      "instructblip-vicuna-13b": "InstructBLIP-13B",
                      "mblip-bloomz-7b": "mBLIP-BLOOMZ-7B",
                      "llava-v1.5-7b": "LLaVA-1.5-7B",
                      "llava-v1.5-13b": "LLaVA-1.5-13B",
                      "SpaceLLaVA": "SpaceLLaVA",
                      "GLaMM-FullScope": "GLaMM-FullScope",
                      "internlm-xcomposer2-vl-7b": "XComposer2",
                      "MiniCPM-Llama3-V-2-5": "MiniCPM-V",
                      "GPT-4o": "GPT-4o"}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["comprehensive", "cpp"], required=True)
    parser.add_argument("--cpp", type=str, choices=["convention", "preferredfor", "perspective"])
    args = parser.parse_args()
    os.makedirs("workspace", exist_ok=True)
    if args.mode == "comprehensive":
        for dataset in ["comfort_ball", "comfort_car"]:
            latex_code = ""
            latex_code = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccccccccc}\n\\hline\n"
            latex_code += "Model & F1 & $soft err_{gt}$ & clipped $err_{gt}$ & hard $err_{gt}$ & $acc$ & $err_{sym.spa}$ & $err_{sym.rr}$ & $noise$ & $p.std$ \\\\ \n\\hline\n"
            metrics_list_obj_hall = []
            metrics_list_spatial = []
            model_names = []
            perspectives = []
            if dataset == "comfort_ball":
                p_cam_errgt_lists = {}
                for perspective in ["camera3"]:
                    results_root = f"results/{dataset}/{perspective}"
                    for _, _, files in os.walk(results_root):
                        for file in files:
                            if file.endswith(".json"):
                                file_path = os.path.join(results_root, file)
                                metrics = spatial_eval(file_path, hard_metrics=False)
                                model_name = file.split(".json")[0].replace("_", "-")
                                metrics_list_obj_hall.append([metrics["mean_yes_positive"], metrics["mean_no_negative"], metrics["mean_yes_negative"], metrics["mean_no_positive"]])
                                metrics_list_spatial.append([metrics["accuracy_metrics_normalized"], metrics["accuracy_metrics_clipped"], metrics["accuracy_metrics_thresholding"], metrics["accuracy_metrics_acc"], metrics["spatial_symmetry_metric_avgs"], metrics["reverse_relation_symmetry_metric_avgs"], metrics["smoothness_metric_avgs"], metrics["std"]])
                                model_names.append(model_name)
                                perspectives.append(perspective)
                    p_cam_errgt_lists[perspective] = []
                min_values = np.min(np.array(metrics_list_spatial), axis=0)
                # print("# metrics for finding minimum:", len(min_values))
                for metrics_obj_hall, metrics_spatial, model_name, perspective in zip(
                    metrics_list_obj_hall,
                    metrics_list_spatial,
                    model_names,
                    perspectives,
                ):
                    # metrics_obj_hall, metrics_spatial, model_name, perspective are metrics per model
                    rounded_metrics = []
                    tp, tn, fp, fn = metrics_obj_hall
                    # print("tp, tn, fp, fn:", tp, tn, fp, fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else tp / 1e-10
                    recall = tp / (tp + fn) if (tp + fn) > 0 else tp / 1e-10
                    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else (2 * precision * recall) / 1e-10
                    # if f1 < 0.5:
                    rounded_metrics.append(f"{f1*100:.1f}")
                    # else:
                    #     rounded_metrics.append(r"\textbf{" + f"{f1*100:.1f}" + "}")
                    p_cam_errgt_lists[perspective].append(metrics_spatial[0])
                    for i, metric in enumerate(metrics_spatial):
                        # print(metric)
                        if round(metric * 100, 1) == round(min_values[i] * 100, 1):
                            rounded_metrics.append(
                                r"\textbf{" + f"{metric*100:.1f}" + "}"
                            )
                        else:
                            rounded_metrics.append(f"{metric*100:.1f}")

                    latex_code += (
                        f"{model_name_mapping[model_name]} ({perspective}) & "
                        + " & ".join(map(str, rounded_metrics))
                        + " \\\\ \n"
                    )
                latex_code += "\\hline\n\\end{tabular}\n\\caption{General metrics on COSINE-Simple}\n\\label{tab:simple_metrics}\n\\end{table}"
                latex_table_B = latex_code
                # print(p_cam_errgt_lists)
                # for perspective in p_cam_errgt_lists.keys():
                #     print(perspective, sum(p_cam_errgt_lists[perspective])/len(p_cam_errgt_lists[perspective]))
            elif dataset == "comfort_car":
                latex_code = ""
                latex_code = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccccccccc}\n\\hline\n"
                latex_code += "Model & F1 & $soft err_{gt}$ & clipped $err_{gt}$ & hard $err_{gt}$ & $acc$ & $err_{sym.spa}$ & $err_{sym.rr}$ & $noise$ & $p.std$ \\\\ \n\\hline\n"
                for perspective in ["camera3"]: # ["nop", "camera3", "reference3", "addressee3"]
                    metrics_list_obj_hall = []
                    metrics_list_spatial = []
                    model_names = []
                    perspectives = []
                    results_root_ref_facing_left = f"results/{dataset}_ref_facing_left/{perspective}"
                    results_root_ref_facing_right = f"results/{dataset}_ref_facing_right/{perspective}"
                    for _, _, files in os.walk(results_root_ref_facing_left):
                        for file in files:
                            if file.endswith(".json"):
                                # This implementation assumes left and right have completely same models!
                                file_path_ref_facing_left = os.path.join(results_root_ref_facing_left, file)
                                file_path_ref_facing_right = os.path.join(results_root_ref_facing_right, file)
                                metrics_ref_facing_left = spatial_eval(
                                    file_path_ref_facing_left, hard_metrics=False, ref_rotation="left"
                                )
                                metrics_ref_facing_right = spatial_eval(
                                    file_path_ref_facing_right, hard_metrics=False, ref_rotation="right"
                                )
                                model_name = file.split(".json")[0].replace(
                                    "_", "-"
                                )
                                metrics_list_obj_hall.append([(metrics_ref_facing_left["mean_yes_positive"] + metrics_ref_facing_right["mean_yes_positive"]) / 2, 
                                                              (metrics_ref_facing_left["mean_no_negative"] + metrics_ref_facing_right["mean_no_negative"]) / 2, 
                                                              (metrics_ref_facing_left["mean_yes_negative"] + metrics_ref_facing_right["mean_yes_negative"]) / 2, 
                                                              (metrics_ref_facing_left["mean_no_positive"] + metrics_ref_facing_right["mean_no_positive"]) / 2
                                                            ])
                                metrics_list_spatial.append([(metrics_ref_facing_left["accuracy_metrics_normalized"] + metrics_ref_facing_right["accuracy_metrics_normalized"]) / 2,
                                                             (metrics_ref_facing_left["accuracy_metrics_clipped"] + metrics_ref_facing_right["accuracy_metrics_clipped"]) / 2, 
                                                             (metrics_ref_facing_left["accuracy_metrics_thresholding"] + metrics_ref_facing_right["accuracy_metrics_thresholding"]) / 2, 
                                                             (metrics_ref_facing_left["accuracy_metrics_acc"] + metrics_ref_facing_right["accuracy_metrics_acc"]) / 2, 
                                                             (metrics_ref_facing_left["spatial_symmetry_metric_avgs"] + metrics_ref_facing_right["spatial_symmetry_metric_avgs"]) / 2, 
                                                             (metrics_ref_facing_left["reverse_relation_symmetry_metric_avgs"] + metrics_ref_facing_right["reverse_relation_symmetry_metric_avgs"]) / 2, 
                                                             (metrics_ref_facing_left["smoothness_metric_avgs"] + metrics_ref_facing_right["smoothness_metric_avgs"]) / 2, 
                                                             (metrics_ref_facing_left["std"] + metrics_ref_facing_right["std"]) / 2
                                                            ])
                                model_names.append(model_name)
                                perspectives.append(perspective)
                    print('-' * 50)
                    min_values = np.min(np.array(metrics_list_spatial), axis=0)
                    # print("# metrics for finding minimum:", len(min_values))
                    for (
                        metrics_obj_hall,
                        metrics_spatial,
                        model_name,
                        perspective,
                    ) in zip(
                        metrics_list_obj_hall,
                        metrics_list_spatial,
                        model_names,
                        perspectives,
                    ):
                        rounded_metrics = []
                        tp, tn, fp, fn = metrics_obj_hall
                        precision = tp / (tp + fp) if (tp + fp) > 0 else tp / 1e-10
                        recall = tp / (tp + fn) if (tp + fn) > 0 else tp / 1e-10
                        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else (2 * precision * recall) / 1e-10
                        # if f1 < 0.5:
                        rounded_metrics.append(f"{f1*100:.1f}")
                        # else:
                        #     rounded_metrics.append(r"\textbf{" + f"{f1*100:.1f}" + "}")
                        for i, metric in enumerate(metrics_spatial):
                            # print(metric)
                            if round(metric * 100, 1) == round(min_values[i] * 100, 1):
                                rounded_metrics.append(
                                    r"\textbf{" + f"{metric*100:.1f}" + "}"
                                )
                            else:
                                rounded_metrics.append(f"{metric*100:.1f}")
                        latex_code += (
                            f"{model_name_mapping[model_name]} ({perspective}) & "
                            + " & ".join(map(str, rounded_metrics))
                            + " \\\\ \n"
                        )
                latex_code += "\\hline\n\\end{tabular}\n\\caption{General metrics on COSINE-Hard}\n\\label{tab:simple_metrics}\n\\end{table}"
                latex_table_C = latex_code
        # print(latex_table_B)
        # print(latex_table_C)
        convert_general_metrics_results(latex_table_B, latex_table_C)
    else:
        assert args.cpp in ["convention", "preferredfor", "perspective"], "In 'cpp' mode, you must specify --cpp to be 'convention' or 'preferredfor' or 'perspective'."
        for cosmode in ["acc", "softcos", "hardcos"]:
            if args.cpp == "perspective":
                ### PERSPECTIVE TAKING
                latex_code = ""
                latex_code = (
                    "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|cccc|cccc|cccc|cccc|}\n\\hline\n"
                )
                latex_code += r"\multirow{2}{*}{Model} & \multicolumn{4}{c|}{behind} & \multicolumn{4}{c|}{infrontof} & \multicolumn{4}{c|}{totheleft} & \multicolumn{4}{c|}{totheright} \\ \cline{2-17} & C & R & A & M & C & R & A & M & C & R & A & M & C & R & A & M \\ \hline"
                latex_code += "\n"
                metrics_list = []
                model_names = []
                dataset = "comfort_car"
                results_root_ref_facing_left = f"results/{dataset}_ref_facing_left"
                results_root_ref_facing_right = f"results/{dataset}_ref_facing_right"
                perspective_metrics_list_for_left_right = []
                perspective_metrics_total_list_for_left_right = []
                for results_root in [results_root_ref_facing_left, results_root_ref_facing_right]:
                    if results_root == f"results/{dataset}_ref_facing_left":
                        model_names, metrics_list = spatial_eval(results_root, hard_metrics=True, cosmode=cosmode, ref_rotation="left")
                    elif results_root == f"results/{dataset}_ref_facing_right":
                        model_names, metrics_list = spatial_eval(results_root, hard_metrics=True, cosmode=cosmode, ref_rotation="right")
                    # print(model_names[0], metrics_list[0])
                    temp_perspective_metrics_list = [] # 3 values
                    temp_perspective_metrics_total_list = [] # adding up 3 values to 1 value
                    for metrics_list_by_model in metrics_list:
                        temp_perspective_metrics_by_configuration = []
                        temp_perspective_metrics_total_by_configuration = []
                        for configuration in metrics_list_by_model.keys():
                            metrics_by_configuration = metrics_list_by_model[configuration]
                            # print(np.mean(np.array(metrics_by_configuration[0]), axis=0).shape)
                            temp_perspective_metrics_by_configuration.append(np.array(metrics_by_configuration[0]))
                            temp_perspective_metrics_total_by_configuration.append(np.sum(np.array(metrics_by_configuration[0]), axis=1))
                        temp_perspective_metrics_list.append(temp_perspective_metrics_by_configuration)
                        temp_perspective_metrics_total_list.append(temp_perspective_metrics_total_by_configuration)
                    # print(temp_perspective_metrics_list)
                    temp_perspective_metrics_list = np.array(temp_perspective_metrics_list)
                    # print(temp_perspective_metrics_list.shape)
                    temp_perspective_metrics_total_list = np.array(temp_perspective_metrics_total_list)
                    # print(temp_perspective_metrics_list.shape, temp_perspective_metrics_total_list.shape)
                    temp_perspective_metrics_list = np.mean(temp_perspective_metrics_list, axis=2)
                    temp_perspective_metrics_total_list = np.mean(temp_perspective_metrics_total_list, axis=2)
                    perspective_metrics_list_for_left_right.append(temp_perspective_metrics_list)
                    perspective_metrics_total_list_for_left_right.append(temp_perspective_metrics_total_list)
                
                perspective_metrics_list = (perspective_metrics_list_for_left_right[0] + perspective_metrics_list_for_left_right[1]) / 2
                perspective_metrics_total_list = (perspective_metrics_total_list_for_left_right[0] + perspective_metrics_total_list_for_left_right[1]) / 2
                min_values = np.min(np.array(perspective_metrics_total_list), axis=0)
                max_values = np.max(np.array(perspective_metrics_total_list), axis=0)
                # print("# metrics for finding minimum:", len(min_values))
                for metrics, model_name in zip(perspective_metrics_list, model_names):
                    rounded_metrics = []
                    # print(f"{model_name}:", (metrics[0][0] + metrics[1][0] + metrics[2][0] + metrics[3][0]) / 4)
                    for i, metric in enumerate(metrics):
                        assert len(metric) == 3, f"length of metric is {len(metric)}"
                        if round((metric[0] + metric[1] + metric[2]) * 100, 1) == round(min_values[i] * 100, 1):
                            if metric[0] == min(metric[0], metric[1], metric[2]):
                                rounded_metrics.append(f"{metric[0]*100:.1f} & {metric[1]*100:.1f} & {metric[2]*100:.1f} & " + f"{((metric[0] + metric[1] + metric[2]) / 3)*100:.1f}")
                            else:
                                rounded_metrics.append(f"{metric[0]*100:.1f} & {metric[1]*100:.1f} & {metric[2]*100:.1f} & " + f"{((metric[0] + metric[1] + metric[2]) / 3)*100:.1f}")
                        # elif round(metric * 100, 1) == round(max_values[i] * 100, 1):
                        #     rounded_metrics.append(
                        #         r"\textbf{\textcolor{red}{" + f"{metric*100:.1f}" + "}}"
                        #     )
                        else:
                            rounded_metrics.append(f"{metric[0]*100:.1f} & {metric[1]*100:.1f} & {metric[2]*100:.1f} & {((metric[0] + metric[1] + metric[2]) / 3)*100:.1f}")
                        pass
                    latex_code += (
                        f"{model_name_mapping[model_name]} & " + " & ".join(map(str, rounded_metrics)) + " \n"
                    )
                latex_code += "\\hline\n\\end{tabular}\n\\caption{COSINE-Hard metric: perspective taking" + f" ({cosmode})" + "}\n\\label{tab:perspective_taking_metrics}\n\\end{table}"
                convert_perspective_taking_results(latex_code, cosmode)
            if args.cpp == "preferredfor":
                ### LANGUAGE AMBIGUITY (Deprecated) NEWNAME: Preferred FOR
                # latex_code = ""
                # latex_code = (
                #     "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|cc|cc|cc|cc|}\n\\hline\n"
                # )
                # latex_code += r"\multirow{2}{*}{Model} & \multicolumn{2}{c|}{behind} & \multicolumn{2}{c|}{infrontof} & \multicolumn{2}{c|}{totheleft} & \multicolumn{2}{c|}{totheright} \\ \cline{2-9} & C & R & C & R & C & R & C & R \\ \hline"
                # latex_code += "\n"
                metrics_list = []
                model_names = []
                dataset = "comfort_car"
                results_root_ref_facing_left = f"results/{dataset}_ref_facing_left"
                results_root_ref_facing_right = f"results/{dataset}_ref_facing_right"
                preferred_for_metrics_list_for_left_and_right = []
                for results_root in [results_root_ref_facing_left, results_root_ref_facing_right]:
                    if results_root == f"results/{dataset}_ref_facing_left":
                        model_names, metrics_list = spatial_eval(results_root, hard_metrics=True, cosmode=cosmode, ref_rotation="left")
                    elif results_root == f"results/{dataset}_ref_facing_right":
                        model_names, metrics_list = spatial_eval(results_root, hard_metrics=True, cosmode=cosmode, ref_rotation="right")
                    temp_preferred_for_metrics_list = []
                    for metrics_list_by_model in metrics_list:
                        preferred_for_metrics_by_configuration = []
                        for configuration in metrics_list_by_model.keys():
                            metrics_by_configuration = metrics_list_by_model[configuration]
                            preferred_for_metrics_by_configuration.append(
                                metrics_by_configuration[1][0] ### NEW: ignore argmin part
                            )
                        temp_preferred_for_metrics_list.append(
                            preferred_for_metrics_by_configuration
                        )
                    preferred_for_metrics_list_for_left_and_right.append(np.array(temp_preferred_for_metrics_list))
                preferred_for_metrics_list = (preferred_for_metrics_list_for_left_and_right[0] + preferred_for_metrics_list_for_left_and_right[0]) / 2
                preferred_for_metrics_list_rel_mean = np.mean(np.array(preferred_for_metrics_list), axis=1)
                preferredfor_data = {
                    ("behind", "C"): [],
                    ("behind", "R"): [],
                    ("behind", "A"): [],
                    ("infrontof", "C"): [],
                    ("infrontof", "R"): [],
                    ("infrontof", "A"): [],
                    ("totheleft", "C"): [],
                    ("totheleft", "R"): [],
                    ("totheleft", "A"): [],
                    ("totheright", "C"): [],
                    ("totheright", "R"): [],
                    ("totheright", "A"): [],
                    ("mean", "C"): [],
                    ("mean", "R"): [],
                    ("mean", "A"): [],
                }
                desired_model_order = ["instructblip-vicuna-7b", "instructblip-vicuna-13b", "mblip-bloomz-7b", "llava-v1.5-7b", "llava-v1.5-13b", "GLaMM-FullScope", "internlm-xcomposer2-vl-7b", "MiniCPM-Llama3-V-2-5", "GPT-4o"]
                model_to_metrics = dict(zip(model_names, preferred_for_metrics_list))
                model_to_metrics_rel_mean = dict(zip(model_names, preferred_for_metrics_list_rel_mean))
                ordered_metrics = [model_to_metrics[model] for model in desired_model_order if model in model_to_metrics]
                ordered_metrics_rel_mean = [model_to_metrics_rel_mean[model] for model in desired_model_order if model in model_to_metrics_rel_mean]
                ordered_model_names = [model for model in desired_model_order if model in model_to_metrics]
                relation_names = ["behind", "infrontof", "totheleft", "totheright"]
                for_names = ["C", "R", "A"]
                index = []
                for metrics, model_name in zip(ordered_metrics, ordered_model_names):
                    index.append(model_name_mapping[model_name])
                    for j, metric in enumerate(metrics):
                        for i in range(0, 3):
                            preferredfor_data[(relation_names[j], for_names[i])].append(f"{metric[i]*100:.1f}")
                for metrics, model_name in zip(ordered_metrics_rel_mean, ordered_model_names):
                    for i in range(0, 3):
                        preferredfor_data[("mean", for_names[i])].append(f"{metrics[i]*100:.1f}")
                df = pd.DataFrame(preferredfor_data, index=index)
                df.to_excel(f"workspace/{args.cpp}_{cosmode}.xlsx", merge_cells=True)
            if args.cpp == "convention":
                convention_data = {
                    ("behind", "Trans"): [],
                    ("behind", "Rot"): [],
                    ("behind", "Refl"): [],
                    ("infrontof", "Trans"): [],
                    ("infrontof", "Rot"): [],
                    ("infrontof", "Refl"): [],
                    ("totheleft", "Trans"): [],
                    ("totheleft", "Rot"): [],
                    ("totheleft", "Refl"): [],
                    ("totheright", "Trans"): [],
                    ("totheright", "Rot"): [],
                    ("totheright", "Refl"): [],
                    ("mean", "Trans"): [],
                    ("mean", "Rot"): [],
                    ("mean", "Refl"): [],
                }
                ### CONVENTION Eval
                latex_code = ""
                latex_code = (
                    "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|ccc|ccc|ccc|ccc|ccc|}\n\\hline\n"
                )
                latex_code += r"\multirow{2}{*}{Model} & \multicolumn{3}{c|}{behind} & \multicolumn{3}{c|}{infrontof} & \multicolumn{3}{c|}{totheleft} & \multicolumn{3}{c|}{totheright} & \multicolumn{3}{c|}{mean} \\ \cline{2-16} & U & R & M & U & R & M & U & R & M & U & R & M & U & R & M \\ \hline"
                latex_code += "\n"
                metrics_list = []
                model_names = []
                dataset = "comfort_ball"
                results_root = f"results/{dataset}"
                # print(spatial_eval(results_root, hard_metrics=True, cosmode=cosmode, convention_eval=True))
                model_names, metrics_list = spatial_eval(results_root, hard_metrics=True, cosmode=cosmode, convention_eval=True)
                convention_metrics_list = [] # 3 values
                for metrics_list_by_model in metrics_list:
                    convention_metrics_by_configuration = []
                    convention_metrics_total_by_configuration = []
                    for configuration in metrics_list_by_model.keys():
                        metrics_by_configuration = metrics_list_by_model[configuration]
                        convention_metrics_by_configuration.append(np.array(metrics_by_configuration))
                    convention_metrics_list.append(convention_metrics_by_configuration)
                convention_metrics_list_original = np.array(convention_metrics_list)
                # print("convention_metrics_list shape:", convention_metrics_list.shape)
                convention_metrics_list = np.mean(convention_metrics_list_original, axis=2)
                convention_metrics_list_total = np.mean(convention_metrics_list_original, axis=(1,2))
                convention_metrics_list_rel_mean = np.mean(np.array(convention_metrics_list), axis=1)
                # min_values = np.min(np.array(convention_metrics_list), axis=0)
                # max_values = np.max(np.array(convention_metrics_list), axis=0)
                # print("# metrics for finding minimum:", len(min_values))
                desired_model_order = ["instructblip-vicuna-7b", "instructblip-vicuna-13b", "mblip-bloomz-7b", "llava-v1.5-7b", "llava-v1.5-13b", "GLaMM-FullScope", "internlm-xcomposer2-vl-7b", "MiniCPM-Llama3-V-2-5", "GPT-4o"]
                model_to_metrics = dict(zip(model_names, convention_metrics_list))
                model_to_metrics_rel_mean = dict(zip(model_names, convention_metrics_list_rel_mean))
                ordered_metrics = [model_to_metrics[model] for model in desired_model_order if model in model_to_metrics]
                ordered_metrics_rel_mean = [model_to_metrics_rel_mean[model] for model in desired_model_order if model in model_to_metrics_rel_mean]
                ordered_model_names = [model for model in desired_model_order if model in model_to_metrics]
                relation_names = ["behind", "infrontof", "totheleft", "totheright"]
                convention_names = ["Trans", "Rot", "Refl"]
                index = []
                for metrics, model_name in zip(ordered_metrics, ordered_model_names):
                    index.append(model_name_mapping[model_name])
                    for j, metric in enumerate(metrics):
                        for i in range(0, 3):
                            convention_data[(relation_names[j], convention_names[i])].append(f"{metric[i]*100:.1f}")
                for metrics, model_name in zip(ordered_metrics_rel_mean, ordered_model_names):
                    for i in range(0, 3):
                        convention_data[("mean", convention_names[i])].append(f"{metrics[i]*100:.1f}")
                df = pd.DataFrame(convention_data, index=index)
                df.to_excel(f"workspace/{args.cpp}_{cosmode}.xlsx", merge_cells=True)

                """
                if cosmode == "acc":
                    if metric == np.max(metrics):
                        metric_string += "\\underline{" + f"{metric*100:.1f}" + '}'
                    else:
                        metric_string += f"{metric*100:.1f}"
                else:
                    if metric == np.min(metrics):
                        metric_string += "\\underline{" + f"{metric*100:.1f}" + '}'
                    else:
                        metric_string += f"{metric*100:.1f}"
                if i == 0 or i == 1:
                    metric_string += " & "
                """

                # for j, (metrics, model_name) in enumerate(zip(ordered_metrics, ordered_model_names)):
                #     rounded_metrics = []
                #     # print(f"{model_name}:", (metrics[0][0] + metrics[1][0] + metrics[2][0] + metrics[3][0]) / 4)
                #     # print(metrics.shape)
                #     rounded_metrics = []
                #     for i, metric in enumerate(metrics):
                #         metric_string = ""
                #         if cosmode == "acc":
                #             if round(metric[0]*100, 1) == max(round(metric[0]*100, 1), round(metric[1]*100, 1), round(metric[2]*100, 1)):
                #                 metric_string += "\\underline{" + f"{metric[0]*100:.1f}" + "} & "
                #             else:
                #                 metric_string += f"{metric[0]*100:.1f}" + " & "
                #             if round(metric[1]*100, 1) == max(round(metric[0]*100, 1), round(metric[1]*100, 1), round(metric[2]*100, 1)):
                #                 metric_string += "\\underline{" + f"{metric[1]*100:.1f}" + "} & "
                #             else:
                #                 metric_string += f"{metric[1]*100:.1f}" + " & "
                #             if round(metric[2]*100, 1) == max(round(metric[0]*100, 1), round(metric[1]*100, 1), round(metric[2]*100, 1)):
                #                 metric_string += "\\underline{" + f"{metric[2]*100:.1f}" + '}'
                #             else:
                #                 metric_string += f"{metric[2]*100:.1f}"
                #         else:
                #             if round(metric[0]*100, 1) == min(round(metric[0]*100, 1), round(metric[1]*100, 1), round(metric[2]*100, 1)):
                #                 metric_string += "\\underline{" + f"{metric[0]*100:.1f}" + "} & "
                #             else:
                #                 metric_string += f"{metric[0]*100:.1f}" + " & "
                #             if round(metric[1]*100, 1) == min(round(metric[0]*100, 1), round(metric[1]*100, 1), round(metric[2]*100, 1)):
                #                 metric_string += "\\underline{" + f"{metric[1]*100:.1f}" + "} & "
                #             else:
                #                 metric_string += f"{metric[1]*100:.1f}" + " & "
                #             if round(metric[2]*100, 1) == min(round(metric[0]*100, 1), round(metric[1]*100, 1), round(metric[2]*100, 1)):
                #                 metric_string += "\\underline{" + f"{metric[2]*100:.1f}" + '}'
                #             else:
                #                 metric_string += f"{metric[2]*100:.1f}"
                #         metric_string += " & "
                #         if i == 3:
                #             for k in range(0, 3):
                #                 if cosmode == "acc":
                #                     if ordered_metrics_total[j][k] == np.max(ordered_metrics_total[j]):
                #                         metric_string += "\\underline{" + f"{ordered_metrics_total[j][k]*100:.1f}" + '}'
                #                     else:
                #                         metric_string += f"{ordered_metrics_total[j][k]*100:.1f}"
                #                 else:
                #                     if ordered_metrics_total[j][k] == np.min(ordered_metrics_total[j]):
                #                         metric_string += "\\underline{" + f"{ordered_metrics_total[j][k]*100:.1f}" + '}'
                #                     else:
                #                         metric_string += f"{ordered_metrics_total[j][k]*100:.1f}"
                #                 if k == 0 or k == 1:
                #                     metric_string += " & "    
                #         rounded_metrics.append(metric_string)
                #     latex_code += (
                #         f"{model_name_mapping[model_name]} & " + " ".join(map(str, rounded_metrics)) + " \\\\ \n"
                #     )
                # latex_code += "\\hline\n\\end{tabular}\n\\caption{COSINE-Hard metric: Convention" + f" ({cosmode})" + "}\n\\label{tab:convention_metrics}\n\\end{table}"
                # print(latex_code)