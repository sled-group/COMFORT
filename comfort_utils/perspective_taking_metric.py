# from .language_ambiguity_metric import extract_data_normalize_and_get_mean
from .accuracy_metric import accuracy_metric
from .helper import FOR_MAP

def perspective_taking_metric(
    results_camera, results_reference, results_addressee, configuration, variation, cosmode, ref_rotation
):
    if cosmode == "softcos":
        cam_acc_normalize = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["camera"][configuration],
            mode="normalize"
        )
        ref_acc_normalize = accuracy_metric(
            results_reference[configuration]["data"][variation]["positive"],
            config=results_reference[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["object"][configuration],
            mode="normalize"
        )
        add_acc_normalize = accuracy_metric(
            results_addressee[configuration]["data"][variation]["positive"],
            config=results_addressee[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["addressee"][configuration],
            mode="normalize"
        )
        return [
            cam_acc_normalize,
            ref_acc_normalize,
            add_acc_normalize,
        ]
    elif cosmode == "hardcos":
        cam_acc_thresholding = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["camera"][configuration],
            mode="thresholding"
        )
        ref_acc_thresholding = accuracy_metric(
            results_reference[configuration]["data"][variation]["positive"],
            config=results_reference[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["object"][configuration],
            mode="thresholding"
        )
        add_acc_thresholding = accuracy_metric(
            results_addressee[configuration]["data"][variation]["positive"],
            config=results_addressee[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["addressee"][configuration],
            mode="thresholding"
        )
        return [
            cam_acc_thresholding,
            ref_acc_thresholding,
            add_acc_thresholding,
        ]
    elif cosmode == "acc":
        cam_acc_acc = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["camera"][configuration],
            mode="acc",
            normalize=False
        )
        ref_acc_acc = accuracy_metric(
            results_reference[configuration]["data"][variation]["positive"],
            config=results_reference[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["object"][configuration],
            mode="acc",
            normalize=False
        )
        add_acc_acc = accuracy_metric(
            results_addressee[configuration]["data"][variation]["positive"],
            config=results_addressee[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["addressee"][configuration],
            mode="acc",
            normalize=False
        )
        return [
            cam_acc_acc,
            ref_acc_acc,
            add_acc_acc,
        ]
    else:
        raise NotImplementedError("This mode for perspective taking metric is not supported yet.")


# cam_acc_clipped = accuracy_metric(
#         results_camera[configuration]["data"][variation]["positive"],
#         config=results_camera[configuration]["data"][variation]["config"],
#         FoR=FOR_MAP["camera"][configuration],
#         mode="clipped",
#     )
# ref_acc_clipped = accuracy_metric(
#     results_reference[configuration]["data"][variation]["positive"],
#     config=results_reference[configuration]["data"][variation]["config"],
#     FoR=FOR_MAP["object"][configuration],
#     mode="clipped",
# )
# add_acc_clipped = accuracy_metric(
#     results_addressee[configuration]["data"][variation]["positive"],
#     config=results_addressee[configuration]["data"][variation]["config"],
#     FoR=FOR_MAP["addressee"][configuration],
#     mode="clipped",
# )