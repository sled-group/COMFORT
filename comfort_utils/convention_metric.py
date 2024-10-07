# from .language_ambiguity_metric import extract_data_normalize_and_get_mean
from .accuracy_metric import accuracy_metric
from .helper import FOR_MAP
import numpy as np

def convention_metric(
    results_camera, configuration, variation, cosmode, ref_rotation
):
    if cosmode == "softcos":
        unrot_cam_acc_normalize = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["unrotated_camera"][configuration],
            mode="normalize"
        )
        rot_cam_acc_normalize = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["rotated_camera"][configuration],
            mode="normalize"
        )
        mixed_cam_acc_normalize = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["mixed_camera"][configuration],
            mode="normalize"
        )
        return [
            unrot_cam_acc_normalize,
            rot_cam_acc_normalize,
            mixed_cam_acc_normalize,
        ]
    elif cosmode == "hardcos":
        unrot_cam_acc_thresholding = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["unrotated_camera"][configuration],
            mode="thresholding"
        )
        rot_cam_acc_thresholding = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["rotated_camera"][configuration],
            mode="thresholding"
        )
        mixed_cam_acc_thresholding = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["mixed_camera"][configuration],
            mode="thresholding"
        )
        return [
            unrot_cam_acc_thresholding,
            rot_cam_acc_thresholding,
            mixed_cam_acc_thresholding,
        ]
    elif cosmode == "acc":
        unrot_cam_acc_acc = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["unrotated_camera"][configuration],
            mode="acc",
            normalize=False
        )
        rot_cam_acc_acc = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["rotated_camera"][configuration],
            mode="acc",
            normalize=False
        )
        mixed_cam_acc_acc = accuracy_metric(
            results_camera[configuration]["data"][variation]["positive"],
            config=results_camera[configuration]["data"][variation]["config"],
            FoR=FOR_MAP[ref_rotation]["mixed_camera"][configuration],
            mode="acc",
            normalize=False
        )
        return [
            unrot_cam_acc_acc,
            rot_cam_acc_acc,
            mixed_cam_acc_acc,
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