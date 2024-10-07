import json
import statistics
import pandas as pd
import numpy as np

preferredfor_data_ratio_avg = {
    "RotC:I": [],
    "TransC:I": [],
    "ReflC:I": [],
    "RotA:I": [],
    "TransA:I": [],
    "ReflA:I": []
}

preferredfor_data_avg = {
    "I": [],
    "RotC": [],
    "TransC": [],
    "ReflC": [],
    "RotA": [],
    "TransA": [],
    "ReflA": []
}

cosmode = "soft"

with open(f'multilingual_preferredfor_raw_{cosmode}_left.json', 'r') as f:
    data_left = json.load(f)

with open(f'multilingual_preferredfor_raw_{cosmode}_right.json', 'r') as f:
    data_right = json.load(f)

filtered_merged_multilingual_preferredfor_raw = {}

average_ratios = {}
std_ratios = {}

raw = {}

langs_not_included = []
langs_evaluated = []

lang_codes = []
for (lang_code_left, values_left), (lang_code_right, values_right) in zip(data_left.items(), data_right.items()):
    assert lang_code_left == lang_code_right, "language code is not aligned"
    if not (len(values_left["rotated_camera_relative"]) == 40 
            and len(values_left["translated_camera_relative"]) == 40 
            and len(values_left["reflected_camera_relative"]) == 40 
            and len(values_left["rotated_addressee_relative"]) == 40
            and len(values_left["translated_addressee_relative"]) == 40
            and len(values_left["reflected_addressee_relative"]) == 40
            and len(values_left["intrinsic"]) == 40
            and len(values_right["rotated_camera_relative"]) == 40 
            and len(values_right["translated_camera_relative"]) == 40 
            and len(values_right["reflected_camera_relative"]) == 40 
            and len(values_right["rotated_addressee_relative"]) == 40
            and len(values_right["translated_addressee_relative"]) == 40
            and len(values_right["reflected_addressee_relative"]) == 40
            and len(values_right["intrinsic"]) == 40):
        langs_not_included.append(lang_code_left)
        continue
    langs_evaluated.append(lang_code_left)
    filtered_merged_multilingual_preferredfor_raw[lang_code_left] = {}
    for p_c in values_left:
        filtered_merged_multilingual_preferredfor_raw[lang_code_left][p_c] = values_left[p_c] + values_right[p_c]
        assert len(filtered_merged_multilingual_preferredfor_raw[lang_code_left][p_c]) == 80
    lang_codes.append(lang_code_left.lower())
    rotated_camera_relative = (np.array(values_left["rotated_camera_relative"]) + np.array(values_right["rotated_camera_relative"])) / 2
    translated_camera_relative = (np.array(values_left["translated_camera_relative"]) + np.array(values_right["translated_camera_relative"])) / 2
    reflected_camera_relative = (np.array(values_left["reflected_camera_relative"]) + np.array(values_right["reflected_camera_relative"])) / 2
    rotated_addressee_relative = (np.array(values_left["rotated_addressee_relative"]) + np.array(values_right["rotated_addressee_relative"])) / 2
    translated_addressee_relative = (np.array(values_left["translated_addressee_relative"]) + np.array(values_right["translated_addressee_relative"])) / 2
    reflected_addressee_relative = (np.array(values_left["reflected_addressee_relative"]) + np.array(values_right["reflected_addressee_relative"])) / 2
    intrinsic = (np.array(values_left["intrinsic"]) + np.array(values_right["intrinsic"])) / 2
    
    rotated_camera_relative_to_intrinsic_ratios = [c / r if r != 0 else 1 for c, r in zip(rotated_camera_relative, intrinsic)]
    translated_camera_relative_to_intrinsic_ratios = [c / r if r != 0 else 1 for c, r in zip(translated_camera_relative, intrinsic)]
    reflected_camera_relative_to_intrinsic_ratios = [c / r if r != 0 else 1 for c, r in zip(reflected_camera_relative, intrinsic)]
    rotated_addressee_relative_to_intrinsic_ratios = [c / r if r != 0 else 1 for c, r in zip(rotated_addressee_relative, intrinsic)]
    translated_addressee_relative_to_intrinsic_ratios = [c / r if r != 0 else 1 for c, r in zip(translated_addressee_relative, intrinsic)]
    reflected_addressee_relative_to_intrinsic_ratios = [c / r if r != 0 else 1 for c, r in zip(reflected_addressee_relative, intrinsic)]
    
    preferredfor_data_ratio_avg["RotC:I"].append(sum(rotated_camera_relative_to_intrinsic_ratios) / len(rotated_camera_relative_to_intrinsic_ratios))
    preferredfor_data_ratio_avg["TransC:I"].append(sum(translated_camera_relative_to_intrinsic_ratios) / len(translated_camera_relative_to_intrinsic_ratios))
    preferredfor_data_ratio_avg["ReflC:I"].append(sum(reflected_camera_relative_to_intrinsic_ratios) / len(reflected_camera_relative_to_intrinsic_ratios))
    preferredfor_data_ratio_avg["RotA:I"].append(sum(rotated_addressee_relative_to_intrinsic_ratios) / len(rotated_addressee_relative_to_intrinsic_ratios))
    preferredfor_data_ratio_avg["TransA:I"].append(sum(translated_addressee_relative_to_intrinsic_ratios) / len(translated_addressee_relative_to_intrinsic_ratios))
    preferredfor_data_ratio_avg["ReflA:I"].append(sum(reflected_addressee_relative_to_intrinsic_ratios) / len(reflected_addressee_relative_to_intrinsic_ratios))

    preferredfor_data_avg["I"].append(sum(intrinsic) / len(intrinsic))
    preferredfor_data_avg["RotC"].append(sum(rotated_camera_relative) / len(rotated_camera_relative))
    preferredfor_data_avg["TransC"].append(sum(translated_camera_relative) / len(translated_camera_relative))
    preferredfor_data_avg["ReflC"].append(sum(reflected_camera_relative) / len(reflected_camera_relative))
    preferredfor_data_avg["RotA"].append(sum(rotated_addressee_relative) / len(rotated_addressee_relative))
    preferredfor_data_avg["TransA"].append(sum(translated_addressee_relative) / len(translated_addressee_relative))
    preferredfor_data_avg["ReflA"].append(sum(reflected_addressee_relative) / len(reflected_addressee_relative))

df = pd.DataFrame(preferredfor_data_ratio_avg, index=lang_codes)
df.to_excel(f"avg_ratio_rel_to_int_{cosmode}.xlsx", merge_cells=True)

df = pd.DataFrame(preferredfor_data_avg, index=lang_codes)
df.to_excel(f"avg_{cosmode}.xlsx", merge_cells=True)

print("Number of languages not included:", len(langs_not_included))
print("Languages not included:", langs_not_included)
print("Number of languages evaluated:", len(langs_evaluated))
print("Number of languages:", len(langs_not_included) + len(langs_evaluated))

with open(f"filtered_merged_multilingual_preferredfor_raw_{cosmode}.json", 'w') as f:
    json.dump(filtered_merged_multilingual_preferredfor_raw, f, indent=4)