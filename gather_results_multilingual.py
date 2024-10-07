import os
import json
import numpy as np
from tqdm import tqdm
from comfort_utils.model_utils.models_api import query_translation, query_translation_back_to_en
from comfort_utils.helper import PERSPECTIVE_PROMPT_MAP, FOR_MAP

model = "gpt-4o"
DEEPL_SUPPORTED_LANGUAGES = ["AR", "BG", "CS", "DA", "DE", "EL", "EN-GB", "EN-US", "ES", "ET", "FI", "FR", "HU", "ID", "IT", "JA", "KO", "LT", "LV", "NB", "NL", "PL", "PT-BR", "PT-PT", "RO", "RU", "SK", "SL", "SV", "TR", "UK", "ZH"]
GOOGLET_SUPPORTED_LANGUAGES = {
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Armenian": "hy",
    "Assamese": "as",
    "Aymara": "ay",
    "Azerbaijani": "az",
    "Bambara": "bm",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bhojpuri": "bho",
    "Bosnian": "bs",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Corsican": "co",
    "Croatian": "hr",
    "Dhivehi": "dv",
    "Dogri": "doi",
    "Esperanto": "eo",
    "Ewe": "ee",
    "Filipino (Tagalog)": "fil",
    "Frisian": "fy",
    "Galician": "gl",
    "Georgian": "ka",
    "Guarani": "gn",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hausa": "ha",
    "Hawaiian": "haw",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hmong": "hmn",
    "Icelandic": "is",
    "Igbo": "ig",
    "Ilocano": "ilo",
    "Irish": "ga",
    "Javanese": "jv",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Khmer": "km",
    "Kinyarwanda": "rw",
    "Konkani": "gom",
    "Krio": "kri",
    "Kurdish": "ku",
    "Kurdish (Sorani)": "ckb",
    "Kyrgyz": "ky",
    "Lao": "lo",
    "Latin": "la",
    "Lingala": "ln",
    "Luganda": "lg",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Maithili": "mai",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Maori": "mi",
    "Marathi": "mr",
    "Meiteilon (Manipuri)": "mni-Mtei",
    "Mizo": "lus",
    "Mongolian": "mn",
    "Myanmar (Burmese)": "my",
    "Nepali": "ne",
    "Nyanja (Chichewa)": "ny",
    "Odia (Oriya)": "or",
    "Oromo": "om",
    "Pashto": "ps",
    "Persian": "fa",
    "Punjabi": "pa",
    "Quechua": "qu",
    "Samoan": "sm",
    "Sanskrit": "sa",
    "Scots Gaelic": "gd",
    "Sepedi": "nso",
    "Serbian": "sr",
    "Sesotho": "st",
    "Shona": "sn",
    "Sindhi": "sd",
    "Sinhala (Sinhalese)": "si",
    "Somali": "so",
    "Sundanese": "su",
    "Swahili": "sw",
    "Tajik": "tg",
    "Tamil": "ta",
    "Tatar": "tt",
    "Telugu": "te",
    "Thai": "th",
    "Tigrinya": "ti",
    "Tsonga": "ts",
    "Turkmen": "tk",
    "Twi (Akan)": "ak",
    "Urdu": "ur",
    "Uyghur": "ug",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Zulu": "zu"
}
SUPPORTED_LANGUAGES = DEEPL_SUPPORTED_LANGUAGES + list(GOOGLET_SUPPORTED_LANGUAGES.values())
# EXCLUDED_LANGUAGES = ["gn", "mni-Mtei", "dv", "ee", "lus", "or", "ti", "ug", "sm", "hr", "xh", "ky", "hmn", "yo", "bm", "ay", "tt", "ml"] # those not following instructions
# SUPPORTED_LANGUAGES = [language for language in SUPPORTED_LANGUAGES if language not in EXCLUDED_LANGUAGES]

translated_yes_tokens_en = ["yes", "Yes", "YES", "Yeah", "yes.", "Yes.", "YES.", "Yeah.", "Oh yes.", "Correct", "Correct.", "Oh yeah.", "Oh yeah"]
translated_no_tokens_en = ["no", "No", "NO", "no.", "No.", "NO.", "not.", "Not.", "NOT.", "Are not", "Are not.", "ARE NOT.", " ARE NOT"]

def belong_to_yes(yes):
    if yes in translated_yes_tokens_en:
        return True
    yes = yes[:5]
    for str in translated_yes_tokens_en:
        if str in yes:
            return True
    return False

def belong_to_no(no):
    if no in translated_no_tokens_en:
        return True
    no = no[:5]
    for str in translated_no_tokens_en:
        if str in no:
            return True
    return False

for ref_rotation in ["left", "right"]:
    if ref_rotation == "left":
        dataset = "comfort_car_ref_facing_left"
    elif ref_rotation == "right":
        dataset = "comfort_car_ref_facing_right"
    for cosmode in ["soft"]:
        ####### Preparing gt_query #######
        gt_query = {}
        for_shift = {
            "camera": 0,
            "addressee": 90,
            "rotated_camera": 180,
            "rotated_addressee": 270,
            "object_facing_right": 90,
            "object_facing_left": 270,
        }
        for gt_cosmode in ["soft", "hard"]:
            gt_query[gt_cosmode] = {}
            for gt_convention in ["mixed", "not_rotated", "rotated"]:
                gt_query[gt_cosmode][gt_convention] = {}
                for gt_perspective in ["camera", "addressee", "object"]:
                    gt_query[gt_cosmode][gt_convention][gt_perspective] = {}
                    for gt_relation in ["infrontof", "behind", "totheleft", "totheright"]:
                        gt_arr = []
                        for angle in [-180, -90, 0, 90]:
                            if gt_convention == "not_rotated":
                                query_gt_convention ="unrotated"
                            else:
                                query_gt_convention =gt_convention
                            shift = for_shift[FOR_MAP[ref_rotation][f"{query_gt_convention}_{gt_perspective}"][gt_relation]]
                            cosine = np.cos((angle + shift) / 180 * np.pi)
                            if np.abs(cosine) < 1e-10:
                                cosine = 0
                            if gt_cosmode == "soft":
                                gt = (cosine + 1) / 2
                            elif gt_cosmode == "hard":
                                gt = np.zeros_like(cosine)
                                gt[cosine > 0] = 1
                                gt = gt.item()
                            gt_arr.append(gt)
                        gt_query[gt_cosmode][gt_convention][gt_perspective][gt_relation] = gt_arr
        # print("gt_query:", gt_query)
        # json.dump(gt_query, open("gt_query.json", 'w'), indent=4)
        # json1 = json.load(open("gt_query.json", 'r'))
        # json2 = json.load(open("gt_closed_source.json", 'r'))
        # print(json1 == json2)
        ####### Preparing gt_query #######

        num_not_found_dict = {}
        num_not_found = 0
        total = 0
        problematic_back_translations = {}
        # preferredfor EVALUATION
        preferredfor_evaluation = {}
        preferredfor_evaluation_raw = {}
        for language in tqdm(SUPPORTED_LANGUAGES):
            preferredfor_evaluation[language] = {}
            preferredfor_evaluation_raw[language] = {}
            num_not_found_dict[language] = 0
            if language not in problematic_back_translations:
                problematic_back_translations[language] = {}
            for perspective in tqdm(["camera3", "reference3", "addressee3"]): # , "addressee3"]):
                if perspective == "camera3":
                    preferredfor_evaluation_raw[language]["rotated_camera_relative"] = []
                    preferredfor_evaluation_raw[language]["translated_camera_relative"] = []
                    preferredfor_evaluation_raw[language]["reflected_camera_relative"] = []
                elif perspective == "addressee3":
                    preferredfor_evaluation_raw[language]["rotated_addressee_relative"] = []
                    preferredfor_evaluation_raw[language]["translated_addressee_relative"] = []
                    preferredfor_evaluation_raw[language]["reflected_addressee_relative"] = []
                elif perspective == "reference3":
                    preferredfor_evaluation_raw[language]["intrinsic"] = []
                results_root_nop = f"results/multilingual/{dataset}/nop/{language}"
                # results_root = f"results/multilingual/{dataset}/{perspective}/{language}"
                file_path_nop = os.path.join(results_root_nop, f"{model}.json")
                # file_path = os.path.join(results_root, f"{model}.json")
                with open(file_path_nop, 'r') as file:
                    all_results_nop = json.load(file)
                # with open(file_path, 'r') as file:
                #     all_results = json.load(file)
                # all_results.pop("dataset_type")
                # all_results.pop("model")
                eval_all_configuration_by_convention = {}
                eval_all_configuration_by_convention["rotated_camera_relative"] = []
                eval_all_configuration_by_convention["translated_camera_relative"] = []
                eval_all_configuration_by_convention["reflected_camera_relative"] = []
                eval_all_configuration_by_convention["rotated_addressee_relative"] = []
                eval_all_configuration_by_convention["translated_addressee_relative"] = []
                eval_all_configuration_by_convention["reflected_addressee_relative"] = []
                eval_all_configuration_by_convention["intrinsic"] = []
                error_per_convention_total = 0
                num_valid_data_perspective = 0
                all_results_nop.pop("dataset_type")
                all_results_nop.pop("model")
                for configuration in all_results_nop.keys():
                    results_by_spatial_rel_nop = all_results_nop[configuration]["data"]
                    # results_by_spatial_rel = all_results[configuration]["data"]
                    error_per_config_total = 0
                    num_valid_data_config = 0
                    for variation in results_by_spatial_rel_nop.keys():
                        results_by_spatial_rel_per_var_nop = results_by_spatial_rel_nop[variation]["positive"]
                        # results_by_spatial_rel_per_var = results_by_spatial_rel[variation]["positive"]
                        results_by_spatial_rel_per_var_vector_nop = []
                        has_missing_data = False
                        for dict_data in results_by_spatial_rel_per_var_nop:
                            yes_no_response = dict_data["response"]["choices"][0]["message"]["content"]
                            logprobs = dict_data["response"]["choices"][0]["logprobs"]
                            yes_no_response_en = query_translation_back_to_en(yes_no_response, language)
                            if belong_to_yes(yes_no_response_en):
                                if logprobs:
                                    top_logprob = logprobs["content"][0]["top_logprobs"]
                                    yes_prob = np.exp(top_logprob[0]["logprob"])
                                    no_prob = np.exp(top_logprob[1]["logprob"])
                                    # print("yes_no_response:", yes_no_response)
                                    # print("Yes prob:", yes_prob)
                                    # print("No prob:", no_prob)
                                    # print("Sum:", yes_prob + no_prob)
                                    normalized_yes_prob = yes_prob / (yes_prob + no_prob)
                                    results_by_spatial_rel_per_var_vector_nop.append(normalized_yes_prob)
                                else:
                                    results_by_spatial_rel_per_var_vector_nop.append(1)
                            elif belong_to_no(yes_no_response_en):
                                if logprobs:
                                    top_logprob = logprobs["content"][0]["top_logprobs"]
                                    yes_prob = np.exp(top_logprob[1]["logprob"])
                                    no_prob = np.exp(top_logprob[0]["logprob"])
                                    # print("yes_no_response:", yes_no_response)
                                    # print("Yes prob:", yes_prob)
                                    # print("No prob:", no_prob)
                                    # print("Sum:", yes_prob + no_prob)
                                    normalized_yes_prob = yes_prob / (yes_prob + no_prob)
                                    results_by_spatial_rel_per_var_vector_nop.append(normalized_yes_prob)
                                else:
                                    results_by_spatial_rel_per_var_vector_nop.append(0)
                            else:
                                has_missing_data = True
                                num_not_found += 1
                                num_not_found_dict[language] += 1
                                # raise Exception("yes and no not found.")
                                # print("original:", yes_no_response, ";translated:", yes_no_response_en, ";language:", language)
                                problematic_back_translations[language][yes_no_response] = yes_no_response_en
                            total += 1
                        # results_by_spatial_rel_per_var_vector = extract_data(results_by_spatial_rel_per_var)
                        if not has_missing_data: # and results_by_spatial_rel_per_var_vector:
                            if cosmode != "acc":
                                gt_configs = {}
                                if perspective == "camera3":
                                    gt_configs["rotated_camera_relative"] = gt_query[cosmode]["rotated"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                    gt_configs["translated_camera_relative"] = gt_query[cosmode]["not_rotated"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                    gt_configs["reflected_camera_relative"] = gt_query[cosmode]["mixed"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                elif perspective == "addressee3":
                                    gt_configs["rotated_addressee_relative"] = gt_query[cosmode]["rotated"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                    gt_configs["translated_addressee_relative"] = gt_query[cosmode]["not_rotated"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                    gt_configs["reflected_addressee_relative"] = gt_query[cosmode]["mixed"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                elif perspective == "reference3":
                                    gt_configs["intrinsic"] = gt_query[cosmode]["not_rotated"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                for gt_type in gt_configs.keys():
                                    gt_config = gt_configs[gt_type]
                                    error = 0
                                    # for data_i in range(0, len(results_by_spatial_rel_per_var_vector)):
                                    #     error += (results_by_spatial_rel_per_var_vector[data_i] - results_by_spatial_rel_per_var_vector_nop[data_i]) ** 2
                                    # error_per_config_total += np.sqrt(error / len(results_by_spatial_rel_per_var_vector))
                                    max_prob = max(results_by_spatial_rel_per_var_vector_nop)
                                    min_prob = min(results_by_spatial_rel_per_var_vector_nop)
                                    for data_i in range(0, len(results_by_spatial_rel_per_var_vector_nop)):
                                        normalized_prob = (results_by_spatial_rel_per_var_vector_nop[data_i] - min_prob) / (max_prob - min_prob)
                                        error += (normalized_prob - gt_config[data_i]) ** 2
                                    preferredfor_evaluation_raw[language][gt_type].append(np.sqrt(error / len(results_by_spatial_rel_per_var_vector_nop)))
                                    error_per_config_total += np.sqrt(error / len(results_by_spatial_rel_per_var_vector_nop))
                                    num_valid_data_config += 1
                            else:
                                gt_configs = {}
                                if perspective == "camera3":
                                    gt_configs["rotated_camera_relative"] = gt_query["hard"]["rotated"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                    gt_configs["translated_camera_relative"] = gt_query["hard"]["not_rotated"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                    gt_configs["reflected_camera_relative"] = gt_query["hard"]["mixed"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                elif perspective == "addressee3":
                                    gt_configs["rotated_addressee_relative"] = gt_query["hard"]["rotated"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                    gt_configs["translated_addressee_relative"] = gt_query["hard"]["not_rotated"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                    gt_configs["reflected_addressee_relative"] = gt_query["hard"]["mixed"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                elif perspective == "reference3":
                                    gt_configs["intrinsic"] = gt_query["hard"]["not_rotated"][PERSPECTIVE_PROMPT_MAP[perspective]][configuration]
                                pred_list = []
                                gt_list = []
                                for gt_type in gt_configs.keys():
                                    gt_config = gt_configs[gt_type]
                                    for data_i in range(0, len(results_by_spatial_rel_per_var_vector_nop)):
                                        prob = results_by_spatial_rel_per_var_vector_nop[data_i]
                                        if prob > 0.5:
                                            pred_list.append(1)
                                        else:
                                            pred_list.append(0)
                                        gt_list.append(gt_config[data_i])
                                    preferredfor_evaluation_raw[language][gt_type].append(sum(p == gt for p, gt in zip(pred_list, gt_list)) / len(pred_list))
                    if cosmode != "acc":      
                        if num_valid_data_config != 0:
                            error_per_convention_total += (error_per_config_total / num_valid_data_config)
                            num_valid_data_perspective += 1
                if cosmode != "acc":
                    if num_valid_data_perspective != 0:
                        error_per_convention_avg = error_per_convention_total / num_valid_data_perspective
                        preferredfor_evaluation[language][gt_type] = error_per_convention_avg
                # else:
                #     preferredfor_evaluation[language][perspective] = None
        # print("preferredfor_evaluation:", preferredfor_evaluation)
        # with open(f"results/eval/multilingual_preferredfor_{cosmode}.json", "w") as fp:
        #     json.dump(preferredfor_evaluation, fp, indent=4)
        # print(len(preferredfor_evaluation_raw[SUPPORTED_LANGUAGES[0]]['rotated_camera_relative']))
        # print("problematic back translations:", problematic_back_translations)

        print("preferredfor_evaluation_raw:", preferredfor_evaluation_raw)
        with open(f"results/eval/multilingual_preferredfor_raw_{cosmode}_{ref_rotation}.json", "w") as fp:
            json.dump(preferredfor_evaluation_raw, fp, indent=4)