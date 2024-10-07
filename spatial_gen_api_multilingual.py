import os
from torch.utils.data import DataLoader
import json
from argparse import ArgumentParser
from tqdm import tqdm
from comfort_utils.model_utils.models_api import gpt4o_completion, query_translation
from comfort_utils.data_utils import CLEVR_generic_path
from comfort_utils.helper import get_prompt_template, show_image

num_erroneous_data = 0

PRIORITIZED_LANGUAGES = ["EN-US", "JA", "KO", "ZH", "ha", "ta"]
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
# SUPPORTED_LANGUAGES = PRIORITIZED_LANGUAGES
print("Number of supported languages:", len(SUPPORTED_LANGUAGES))

batch_size = 1

SUPPORTED_MODELS = [
    "gpt-4v",
    "gpt-4o",
]

parser = ArgumentParser()
# parser.add_argument(
#     "--language",
#     type=str,
#     default="EN-US",
#     choices=SUPPORTED_LANGUAGES,
# )
parser.add_argument(
    "--model",
    type=str,
    default="gpt-4o",
    choices=SUPPORTED_MODELS,
)
parser.add_argument("--data_path", type=str, default="data/comfort_car_ref_facing_right")
args = parser.parse_args()
print("Using args: ", args)
dataset_path_root = args.data_path

if args.model == "gpt-4o":
    model = gpt4o_completion
else:
    raise ValueError(f"Model {args.model} not found")

# Use the following dictionaries to convert the dataset configuration to human-readable text
dataset_type = args.data_path.split("_")[1]
dataset_type_full = '_'.join(args.data_path.split('_')[1:])
print("dataset_type full:", dataset_type_full)
for target_lang in SUPPORTED_LANGUAGES:
    for perspective_prompt in ["nop", "camera3", "addressee3", "reference3"]:
        if dataset_type == "ball":
            shape_names = {
                "Sphere": "sphere",
                "SmoothCube_v2": "cube",
                "SmoothCylinder": "cylinder",
                "FatCylinder": "cylinder",
                "Table": "table",
                "Box": "box",
            }
            color_names = ["red", "blue", "green", "yellow"]
            distractor_obj = "green sphere"
            results_root = f"results/multilingual/comfort_{dataset_type_full}/{perspective_prompt}"
        elif dataset_type == "car":
            shape_names = {
                # reference objects
                "Horse": "horse",
                "Bench": "bench",
                "Laptop": "laptop",
                "Sofa": "sofa",
                "Basketball": "basketball",
                "Chair": "chair",
                "Dog": "dog",
                "Bed": "bed",
                "Duck": "duck",
                "bicycle_mountain": "bicycle",
                "car_sedan": "car",

                # addressee
                "Sophia": "woman",
            }
            results_root = f"results/multilingual/comfort_{dataset_type_full}/{perspective_prompt}/{target_lang}"
        else:
            raise ValueError(f"Dataset type not exist: {dataset_type}")
        relation_names = {
            "above": "above",
            "behind": "behind",
            "in": "in",
            "inbetween": "in between",
            "infrontof": "in front of",
            "inthemiddleof": "in the middle of",
            "totheleft": "to the left of",
            "totheright": "to the right of",
            "under": "under",
        }
        x_names = {
            "rotate": "angle",
            "translate": "translation",
        }

        # objects_list = []
        # for shape in shape_names.values():
        #     for color in color_names:
        #         objects_list.append(f"{color} {shape}")

        # obj_hallucination_prompt_template = "Is there any {obj}?"

        positive_prompt_template = get_prompt_template(perspective_prompt)

        model_name = args.model
        results_name = f"{model_name}.json"
        results_path = os.path.join(results_root, results_name)
        if os.path.exists(results_path):
            # Load the JSON data into the result variable
            with open(results_path, "r") as file:
                results = json.load(file)
            print(f"{results_path} loaded successfully.")
        else:
            if not os.path.exists(results_root):
                os.makedirs(results_root)
            results = {}
            print(f"{results_path} does not exist. Setting to empty.")

        results["dataset_type"] = dataset_type
        results["model"] = model_name
        dataset_types = sorted(list(os.listdir(dataset_path_root)))
        configurations_pbar_desc = "Total evaluation progress"
        configurations = tqdm(dataset_types, desc=configurations_pbar_desc)
        for configuration in configurations:
            if not os.path.isdir(os.path.join(dataset_path_root, configuration)):
                # Skip files in the directory
                continue
            # if configuration in results.keys():
            #     continue
            if configuration in results:
                temp_results_configuration = results[configuration]
            else:
                temp_results_configuration = False
            results[configuration] = {}
            relation = relation_names[configuration]
            data = {}

            variation_types = sorted(
                list(os.listdir(os.path.join(dataset_path_root, configuration)))
            )
            variations_pbar_desc = f'Evaluating "{configuration}"'.ljust(
                len(configurations_pbar_desc)
            )
            with tqdm(
                total=len(variation_types) * 1, desc=variations_pbar_desc, leave=False
            ) as variations_pbar:
                for variation_type in variation_types:
                    if not os.path.isdir(
                        os.path.join(dataset_path_root, configuration, variation_type)
                    ):
                        # Skip files in the directory
                        continue
                    data[variation_type] = {}
                    dataset_path = os.path.join(
                        dataset_path_root, f"{configuration}/{variation_type}"
                    )
                    dataset = CLEVR_generic_path(img_dir=dataset_path)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                    shape1 = shape_names[dataset.config["var_shape"]]
                    if dataset_type == "ball":
                        color1 = dataset.config["var_color"]
                        obj1 = f"{color1} {shape1}"
                    elif dataset_type == "car":
                        obj1 = f"{shape1}"

                    shape2 = shape_names[dataset.config["ref_shape"]]
                    if dataset_type == "ball":
                        color2 = dataset.config["ref_color"]
                        obj2 = f"{color2} {shape2}"
                    elif dataset_type == "car":
                        obj2 = f"{shape2}"

                    if perspective_prompt[: len("reference")] != "reference":
                        positive_prompt = positive_prompt_template.format(
                            obj1=obj1, relation=relation, obj2=obj2
                        )
                    else:
                        positive_prompt = positive_prompt_template.format(
                            reference=obj2, obj1=obj1, relation=relation, obj2=obj2
                        )
                    translated_user_prompt = query_translation(positive_prompt, target_lang)
                    system_instruction_prompt =  f'You will be provided an image and a question, please answer the question only in "{query_translation("Yes", target_lang)}" or "{query_translation("No", target_lang)}"'
                    translated_system_instruction_prompt = query_translation(system_instruction_prompt, target_lang)

                    stats_prompt = []
                    positive_data_idx = 0
                    for batch_idx, batch in enumerate(dataloader):
                        if batch_idx in [0, 9, 18, 27]:
                            image_path = batch[0][0]
                            if temp_results_configuration == False:
                                response = model(image_path, translated_user_prompt, translated_system_instruction_prompt)
                                stats_prompt.append({"response": response, "image_path": image_path})
                            elif "response.text" in temp_results_configuration["data"][variation_type]["positive"][positive_data_idx]["response"].keys():
                                # print("batch:", batch)
                                # show_image(image_path)
                                num_erroneous_data += 1
                                print("Found one erroneous data, regenerating... Number of erroneous data:", num_erroneous_data)
                                response = model(image_path, translated_user_prompt, translated_system_instruction_prompt)
                                stats_prompt.append({"response": response, "image_path": image_path})
                            else:
                                stats_prompt.append({"response": temp_results_configuration["data"][variation_type]["positive"][positive_data_idx]["response"], "image_path": image_path})
                            positive_data_idx += 1

                    data[variation_type]["system_prompt_before_translation"] = system_instruction_prompt
                    data[variation_type]["system_prompt_after_translation"] = translated_system_instruction_prompt
                    data[variation_type]["positive_prompt_before_translation"] = positive_prompt
                    data[variation_type]["positive_prompt_after_translation"] = translated_user_prompt
                    data[variation_type]["positive"] = stats_prompt
                    variations_pbar.update(1)

                    data[variation_type]["config"] = dataset.config

            results[configuration]["data"] = data
            results[configuration]["perspective_prompt"] = perspective_prompt
            if perspective_prompt[: len("reference")] != "reference":
                results[configuration]["positive_template"] = positive_prompt_template.format(
                    obj1="[A]", relation=relation, obj2="[B]"
                )
            else:
                results[configuration]["positive_template"] = positive_prompt_template.format(
                    reference="[B]", obj1="[A]", relation=relation, obj2="[B]"
                )
            results[configuration]["x_name"] = x_names[dataset.config["path_type"]]
            results[configuration]["config"] = results[configuration]["data"]["default"][
                "config"
            ]

            with open(results_path, "w") as fp:
                json.dump(results, fp, indent=4)
print("num_erroneous_data:", num_erroneous_data)