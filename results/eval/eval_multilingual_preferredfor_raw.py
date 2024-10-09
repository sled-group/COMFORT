import json
import pandas as pd
import numpy as np

preferredfor_data_ratio_avg = {k: [] for k in ["RotC:I", "TransC:I", "ReflC:I", "RotA:I", "TransA:I", "ReflA:I"]}
preferredfor_data_avg = {k: [] for k in ["I", "RotC", "TransC", "ReflC", "RotA", "TransA", "ReflA"]}

cosmode = "soft"
data_left = json.load(open(f'multilingual_preferredfor_raw_{cosmode}_left.json', 'r'))
data_right = json.load(open(f'multilingual_preferredfor_raw_{cosmode}_right.json', 'r'))

filtered_merged_multilingual_preferredfor_raw = {}
langs_not_included, langs_evaluated, lang_codes = [], [], []

def calculate_avg_ratio(values_left, values_right, key):
    left, right = np.array(values_left[key]), np.array(values_right[key])
    return (left + right) / 2

def ratio_to_intrinsic(arr, intrinsic):
    return [a / i if i != 0 else 1 for a, i in zip(arr, intrinsic)]

key_map = {
    "RotC": "rotated_camera_relative",
    "TransC": "translated_camera_relative",
    "ReflC": "reflected_camera_relative",
    "RotA": "rotated_addressee_relative",
    "TransA": "translated_addressee_relative",
    "ReflA": "reflected_addressee_relative"
}

for (lang_left, values_left), (lang_right, values_right) in zip(data_left.items(), data_right.items()):
    assert lang_left == lang_right, "Language codes are not aligned"
    
    if not all(len(values_left[key]) == 40 and len(values_right[key]) == 40 for key in values_left):
        langs_not_included.append(lang_left)
        continue
    
    langs_evaluated.append(lang_left)
    lang_codes.append(lang_left.lower())
    
    filtered_merged_multilingual_preferredfor_raw[lang_left] = {k: values_left[k] + values_right[k] for k in values_left}
    avg_intrinsic = calculate_avg_ratio(values_left, values_right, "intrinsic")
    
    for prefix, key in key_map.items():
        avg_camera = calculate_avg_ratio(values_left, values_right, key)
        
        preferredfor_data_ratio_avg[f"{prefix}:I"].append(np.mean(ratio_to_intrinsic(avg_camera, avg_intrinsic)))
        
        preferredfor_data_avg[prefix].append(np.mean(avg_camera))
    
    preferredfor_data_avg["I"].append(np.mean(avg_intrinsic))

pd.DataFrame(preferredfor_data_ratio_avg, index=lang_codes).to_excel(f"avg_ratio_rel_to_int_{cosmode}.xlsx", merge_cells=True)
pd.DataFrame(preferredfor_data_avg, index=lang_codes).to_excel(f"avg_{cosmode}.xlsx", merge_cells=True)

with open(f"filtered_merged_multilingual_preferredfor_raw_{cosmode}.json", 'w') as f:
    json.dump(filtered_merged_multilingual_preferredfor_raw, f, indent=4)

print("Number of languages not included:", len(langs_not_included))
print("Languages not included:", langs_not_included)
print("Number of languages evaluated:", len(langs_evaluated))
print("Total languages:", len(langs_not_included) + len(langs_evaluated))