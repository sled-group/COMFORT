from .symmetry_metric import normalize_data
import numpy as np


def accuracy_metric(data, config, FoR, mode="normalize", normalize=True):
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

    if normalize:
        this_data = normalize_data(data)
    else:
        this_data = data
    assert len(this_data) == 37, f"Data length should be 37, got: {len(this_data)}"
    if len(this_data) == 37:
        this_data = this_data[:-1]
    if mode != "acc":
        error = 0
        for _, (angle, likelihoods) in enumerate(this_data):
            shift = for_shift[FoR]
            cosine = np.cos((angle + shift) / 180 * np.pi)
            if np.abs(cosine) < 1e-10:
                cosine = 0
            if mode == "normalize":
                gt = (cosine + 1) / 2
            elif mode == "clipped": # mode == "clipped"
                gt = np.clip(cosine, 0, 1)
            elif mode == "thresholding":
                gt = np.zeros_like(cosine)
                gt[cosine > 0] = 1
            else:
                raise ValueError("Invalid mode")
            error += (gt - likelihoods["Yes"]) ** 2
        return np.sqrt(error / len(data))
    else:
        acc_list = []
        gt_list = []
        for _, (angle, likelihoods) in enumerate(this_data):
            shift = for_shift[FoR]
            cosine = np.cos((angle + shift) / 180 * np.pi)
            if np.abs(cosine) < 1e-10:
                cosine = 0
            # gt = np.zeros_like(cosine)
            # gt[cosine > 0] = 1
            if cosine > 0:
                gt = 1
            else:
                gt = 0
            pred = None
            # print("likelihoods:", likelihoods)
            if likelihoods["Yes"] > 0.5:
                pred = 1
            else:
                pred = 0
            if pred == gt:
                acc_list.append(1)
            else:
                acc_list.append(0)
            gt_list.append(gt)
        # print(len(gt_list))
        # print(gt_list.count(1))
        # print(acc_list.count(1))
        return sum(acc_list) / len(acc_list)
    

def accuracy_metric_old(data, config, perspective=False):
    if config is not None:
        x_type = config["path_type"]
    if x_type == "rotate":
        assert config["relation"] in [
            "infrontof",
            "behind",
            "totheleft",
            "totheright",
        ], "relation must be one of infrontof, behind, totheleft, totheright"
        error = 0
        normalized_data = normalize_data(data)
        for _, (angle, likelihoods) in enumerate(normalized_data):
            if perspective:
                rot = config["ref_rotation"]
                if rot is None:
                    rot = [90, 0, 0]
                if config["relation"] in ["infrontof", "behind"]:
                    shift = -(rot[-1] + 90)
                else:
                    shift = -(rot[-1] - 90)
            else:
                shift = 0
            gt = (np.cos((angle - 180 + shift) / 180 * np.pi) + 1) / 2
            error += (gt - likelihoods["Yes"]) ** 2
            # error += (1 - gt - likelihoods["No"]) ** 2
        return np.sqrt(error / len(data))
    if x_type == "translate":
        assert config is not None, "config must be provided for translation"
        cosines = {}
        for [x_coord, ref_pos, var_pos] in config["mapping"].values():
            ref_pos = np.array(ref_pos)
            var_pos = np.array(var_pos)
            distance = np.sqrt(((ref_pos - var_pos) ** 2).sum())
            if config["relation"] in ["above", "under", "inthemiddleof"]:
                z_dist = abs(config["ref_position"][2] - config["start_point"][2])
                cosines[round(x_coord, 3)] = z_dist / distance
            elif config["relation"] in ["inbetween"]:
                ref1_pos, ref2_pos = config["ref_position"]
                ref1_pos = np.array(ref1_pos)
                ref2_pos = np.array(ref2_pos)
                refs_dist = np.sqrt(((ref1_pos - ref2_pos) ** 2).sum())
                cosines[round(x_coord, 3)] = (refs_dist / 2) / distance
            else:
                raise ValueError("Invalid relation")
        error = 0
        normalized_data = normalize_data(data)
        # Normalize the cosines
        max_cosine = max(cosines.values())
        min_cosine = min(cosines.values())
        for key in cosines:
            cosines[key] = (cosines[key] - min_cosine) / (max_cosine - min_cosine)
        for _, (x, likelihoods) in enumerate(normalized_data):
            # print(cosines[round(x[0], 3)], likelihoods)
            # error += (cosines[round(x[0], 3)] - likelihoods["Yes"]) ** 2
            error += (cosines[round(x, 3)] - likelihoods["Yes"]) ** 2
            # error += (1 - cosines[round(x[0], 3)] - likelihoods["No"]) ** 2
        return np.sqrt(error / len(data))


def accuracy_metric_object_frame(data, config):
    if config is not None:
        x_type = config["path_type"]
    if x_type == "rotate":
        assert config["relation"] in [
            "infrontof",
            "behind",
            "totheleft",
            "totheright",
        ], "relation must be one of infrontof, behind, totheleft, totheright"
        error = 0
        normalized_data = normalize_data(data)
        for _, (angle, likelihoods) in enumerate(normalized_data):
            gt = (np.cos((angle - 180) / 180 * np.pi) + 1) / 2
            error += (gt - likelihoods["Yes"]) ** 2
            # error += (1 - gt - likelihoods["No"]) ** 2
        return np.sqrt(error / len(data))
