from .accuracy_metric import accuracy_metric
from .symmetry_metric import normalize_data
import numpy as np
from .helper import FOR_MAP

### Language Ambiguity Metric is DEPRECATED, but you can still use it~ ###

# def extract_data_normalize_and_get_mean(data_all_variations):
#     all_extracted_data = []
#     for data_variation in data_all_variations:
#         extracted_data = []
#         normalized_data = normalize_data(data_variation)
#         for _, (_, likelihoods) in enumerate(normalized_data):
#             extracted_data.append(likelihoods["Yes"])
#         all_extracted_data.append(np.array(extracted_data))
#     return np.mean(np.vstack(all_extracted_data), axis=0)


# def language_ambiguity_metric(
#     data_nop_all_variations, data_camera_all_variations, data_reference_all_variations, data_addressee_all_variations
# ):
#     normalized_mean_data_nop = extract_data_normalize_and_get_mean(
#         data_nop_all_variations
#     )
#     normalized_mean_data_camera = extract_data_normalize_and_get_mean(
#         data_camera_all_variations
#     )
#     normalized_mean_data_reference = extract_data_normalize_and_get_mean(
#         data_reference_all_variations
#     )
#     normalized_mean_data_addressee = extract_data_normalize_and_get_mean(
#         data_addressee_all_variations
#     )
#     return [np.mean(np.abs(normalized_mean_data_nop - normalized_mean_data_camera)), np.mean(np.abs(normalized_mean_data_nop - normalized_mean_data_reference)), np.mean(np.abs(normalized_mean_data_nop - normalized_mean_data_addressee))]

def language_ambiguity_metric(
    results_nop, configuration, cosmode, ref_rotation
) -> tuple[float, str]:
    perspectives = ["camera", "object", "addressee"]
    acc = []
    if cosmode == "softcos":
        for perspective in perspectives:
            acc_var = []
            for variation in results_nop[configuration]["data"]:
                acc_var.append(
                        accuracy_metric(
                        results_nop[configuration]["data"][variation]["positive"],
                        config=results_nop[configuration]["data"][variation]["config"],
                        FoR=FOR_MAP[ref_rotation][perspective][configuration],
                        mode="normalize"
                    )
                )
            acc.append(np.mean(acc_var))
    elif cosmode == "hardcos":
        for perspective in perspectives:
            acc_var = []
            for variation in results_nop[configuration]["data"]:
                acc_var.append(
                        accuracy_metric(
                        results_nop[configuration]["data"][variation]["positive"],
                        config=results_nop[configuration]["data"][variation]["config"],
                        FoR=FOR_MAP[ref_rotation][perspective][configuration],
                        mode="thresholding"
                    )
                )
            acc.append(np.mean(acc_var))
    elif cosmode == "acc":
        for perspective in perspectives:
            acc_var = []
            for variation in results_nop[configuration]["data"]:
                acc_var.append(
                        accuracy_metric(
                        results_nop[configuration]["data"][variation]["positive"],
                        config=results_nop[configuration]["data"][variation]["config"],
                        FoR=FOR_MAP[ref_rotation][perspective][configuration],
                        mode="acc",
                        normalize=False
                    )
                )
            acc.append(np.mean(acc_var))
    else:
        raise NotImplementedError("This cosmode for language ambiguity metric is not supported yet.")
    preference = np.argmin(acc)
    return acc, preference