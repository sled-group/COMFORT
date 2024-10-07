import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from torch.utils.data import DataLoader
from comfort_utils.model_utils import (
    LlavaWrapper,
    LlavaHfWrapper,
    LlavaGdWrapper,
    BlipWrapper,
    XComposerWrapper,
    MiniCpmWrapper,
    # GLaMMWrapper,
    GptWrapper,
)
from comfort_utils.data_utils import CLEVR_generic
from comfort_utils.helper import prompt_spatial, get_prompt_template
import json
from argparse import ArgumentParser
from tqdm import tqdm

SUPPORTED_MODELS = [
    "liuhaotian/llava-v1.5-7b",
    "liuhaotian/llava-v1.5-13b",
    "Salesforce/blip2-opt-2.7b-coco",
    "Salesforce/blip2-opt-6.7b-coco",
    "Salesforce/instructblip-vicuna-7b",
    "Salesforce/instructblip-vicuna-13b",
    # "Haozhangcx/llava_grounding_gd_vp",
    # "xtuner/llava-llama-3-8b-transformers",
    "internlm/internlm-xcomposer2-vl-7b",
    "openbmb/MiniCPM-Llama3-V-2_5",
    # "OpenGVLab/InternVL-Chat-V1-5",
    "remyxai/SpaceLLaVA",
    "MBZUAI/GLaMM-FullScope",
    "GPT-4o",
    "Gregor/mblip-bloomz-7b",
]

parser = ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="liuhaotian/llava-v1.5-7b",
    choices=SUPPORTED_MODELS,
)
parser.add_argument("--data_path", type=str, default="data/comfort_ball")
parser.add_argument(
    "--perspective_prompt",
    type=str,
    default="camera1",
    choices=[
        "nop",
        "camera1",
        "camera2",
        "camera3",
        "egocentric3",
        "addressee1",
        "addressee2",
        "addressee3",
        "reference1",
        "reference2",
        "reference3",
    ],
)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--layer_wise", action="store_true")
args = parser.parse_args()
print("Using args: ", args)
model_path = args.model
dataset_path_root = args.data_path
batch_size = args.batch_size
quantize = args.quantize
layer_wise = args.layer_wise

if "llava_grounding" in model_path:
    model_wrapper = LlavaGdWrapper(model_path, quantize=quantize)
elif "llava-llama-3" in model_path:
    model_wrapper = LlavaHfWrapper(model_path, quantize=quantize)
elif "llava" in model_path or model_path == "remyxai/SpaceLLaVA":
    model_wrapper = LlavaWrapper(model_path, quantize=quantize)
elif "blip" in model_path:
    model_wrapper = BlipWrapper(model_path, quantize=quantize)
elif "xcomposer2" in model_path:
    model_wrapper = XComposerWrapper(model_path, quantize=quantize)
elif "MiniCPM" in model_path:
    model_wrapper = MiniCpmWrapper(model_path, quantize=quantize)
elif "GLaMM" in model_path:
    model_wrapper = GLaMMWrapper(model_path, quantize=quantize)
elif model_path == "GPT-4o":
    model_wrapper = GptWrapper("gpt-4o")
else:
    raise ValueError(f"Model {model_path} not found")

# Use the following dictionaries to convert the dataset configuration to human-readable text
dataset_type = args.data_path.split("_")[1]
dataset_type_full = '_'.join(args.data_path.split('_')[1:])
print("dataset_type_full:", dataset_type_full)
distractor_obj = None
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
    results_root = f"results/comfort_{dataset_type_full}/{args.perspective_prompt}"
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

        "Sophia": "woman"
    }
    results_root = f"results/comfort_{dataset_type_full}/{args.perspective_prompt}"
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

if dataset_type == "ball":
    objects_list = []
    for shape in shape_names.values():
        for color in color_names:
            objects_list.append(f"{color} {shape}")
elif dataset_type == "car":
    objects_list = []
    for shape in shape_names.values():
        objects_list.append(f"{shape}")

obj_hallucination_prompt_template = "Is there any {obj}?"
positive_prompt_template = get_prompt_template(args.perspective_prompt)

model_name = model_path.split("/")[-1]
if quantize:
    model_name += "_4bit"
if layer_wise:
    model_name += "_layer_wise"
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
results["dataset_type_full"] = dataset_type_full
results["model"] = model_name

spatial_relationship_types = sorted(list(os.listdir(dataset_path_root)))
print("spatial_relationship_types:", spatial_relationship_types)
configurations_pbar_desc = "Total evaluation progress"
configurations = tqdm(spatial_relationship_types, desc=configurations_pbar_desc)
for configuration in configurations:
    if not os.path.isdir(os.path.join(dataset_path_root, configuration)):
        # Skip files in the directory
        continue
    if configuration in results.keys():
        continue
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
        total=len(variation_types) * 3, desc=variations_pbar_desc, leave=False
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
            dataset = CLEVR_generic(img_dir=dataset_path)
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

            # positive_obj1_hallucination_prompt = (
            #     obj_hallucination_prompt_template.format(obj=obj1)
            # )

            positive_obj2_hallucination_prompt = (
                obj_hallucination_prompt_template.format(obj=obj2)
            )

            if dataset_type == "ball":
                negative_objects = [
                    obj
                    for obj in objects_list
                    if obj != obj1 and obj != obj2 and obj != distractor_obj
                ]

                obj1_color = obj1.split()[0]
                obj1_shape = obj1.split()[1]
                obj2_color = obj2.split()[0]
                obj2_shape = obj2.split()[1]
                existing_colors = [obj1_color, obj2_color]
                existing_shapes = [obj1_shape, obj2_shape]
                if distractor_obj != None:
                    distractor_color = distractor_obj.split()[0]
                    distractor_shape = distractor_obj.split()[1]
                    existing_colors.append(distractor_color)
                    existing_shapes.append(distractor_shape)

                for neg_obj in negative_objects:
                    neg_obj_color = neg_obj.split()[0]
                    neg_obj_shape = neg_obj.split()[1]
                    if (
                        neg_obj_color not in existing_colors
                        and neg_obj_shape not in existing_shapes
                    ):
                        negative_color_shape_object = neg_obj
                    if (
                        neg_obj_color not in existing_colors
                        and neg_obj_shape in existing_shapes
                    ):
                        negative_color_object = neg_obj
                    if (
                        neg_obj_color in existing_colors
                        and neg_obj_shape not in existing_shapes
                    ):
                        negative_shape_object = neg_obj
            elif dataset_type == "car":
                negative_objects = [
                    obj
                    for obj in objects_list
                    if obj != obj1 and obj != obj2
                ]

                negative_color_shape_object = negative_objects[0]

            negative_color_shape_object_hallucination_prompt = (
                obj_hallucination_prompt_template.format(
                    obj=negative_color_shape_object
                )
            )

            # negative_color_object_hallucination_prompt = (
            #     obj_hallucination_prompt_template.format(obj=negative_color_object)
            # )

            # negative_shape_object_hallucination_prompt = (
            #     obj_hallucination_prompt_template.format(obj=negative_shape_object)
            # )

            # data[variation_type]["object1_hallucination_positive"] = prompt_spatial(
            #     dataloader=dataloader,
            #     model_wrapper=model_wrapper,
            #     prompt=positive_obj1_hallucination_prompt,
            #     inner_progress=True,
            #     desc=f"{variation_type}/object1_hallucination_positive".ljust(
            #         len(configurations_pbar_desc)
            #     ),
            #     layer_wise=layer_wise,
            # )
            # variations_pbar.update(1)

            data[variation_type]["object2_hallucination_positive"] = prompt_spatial(
                dataloader=dataloader,
                model_wrapper=model_wrapper,
                prompt=positive_obj2_hallucination_prompt,
                inner_progress=True,
                desc=f"{variation_type}/object2_hallucination_positive".ljust(
                    len(configurations_pbar_desc)
                ),
                layer_wise=layer_wise,
            )
            variations_pbar.update(1)

            data[variation_type]["object_hallucination_negative_color_shape"] = (
                prompt_spatial(
                    dataloader=dataloader,
                    model_wrapper=model_wrapper,
                    prompt=negative_color_shape_object_hallucination_prompt,
                    inner_progress=True,
                    desc=f"{variation_type}/object_hallucination_negative_color_shape".ljust(
                        len(configurations_pbar_desc)
                    ),
                    layer_wise=layer_wise,
                )
            )
            variations_pbar.update(1)

            # data[variation_type]["object_hallucination_negative_color"] = (
            #     prompt_spatial(
            #         dataloader=dataloader,
            #         model_wrapper=model_wrapper,
            #         prompt=negative_color_object_hallucination_prompt,
            #         inner_progress=True,
            #         desc=f"{variation_type}/object_hallucination_negative_color".ljust(
            #             len(configurations_pbar_desc)
            #         ),
            #         layer_wise=layer_wise,
            #     )
            # )
            # variations_pbar.update(1)

            # data[variation_type]["object_hallucination_negative_shape"] = (
            #     prompt_spatial(
            #         dataloader=dataloader,
            #         model_wrapper=model_wrapper,
            #         prompt=negative_shape_object_hallucination_prompt,
            #         inner_progress=True,
            #         desc=f"{variation_type}/object_hallucination_negative_shape".ljust(
            #             len(configurations_pbar_desc)
            #         ),
            #         layer_wise=layer_wise,
            #     )
            # )
            # variations_pbar.update(1)

            if args.perspective_prompt[: len("reference")] != "reference":
                positive_prompt = positive_prompt_template.format(
                    obj1=obj1, relation=relation, obj2=obj2
                )
            else:
                positive_prompt = positive_prompt_template.format(
                    reference=obj2, obj1=obj1, relation=relation, obj2=obj2
                )
            data[variation_type]["positive"] = prompt_spatial(
                dataloader=dataloader,
                model_wrapper=model_wrapper,
                prompt=positive_prompt,
                inner_progress=True,
                desc=f"{variation_type}/positive".ljust(len(configurations_pbar_desc)),
                layer_wise=layer_wise,
            )
            variations_pbar.update(1)

            # negative_prompt = negative_prompt_template.format(
            #     obj1=obj1, relation=relation, obj2=obj2
            # )
            # data[variation_type]["negative"] = prompt_spatial(
            #     dataloader,
            #     negative_prompt,
            #     image_processor,
            #     model,
            #     tokenizer,
            #     inner_progress=True,
            #     desc=f"{variation_type}/negative".ljust(len(configurations_pbar_desc)),
            # )

            data[variation_type]["config"] = dataset.config

    results[configuration]["data"] = data
    results[configuration]["perspective_prompt"] = args.perspective_prompt
    if args.perspective_prompt[: len("reference")] != "reference":
        results[configuration]["positive_template"] = positive_prompt_template.format(
            obj1="[A]", relation=relation, obj2="[B]"
        )
    else:
        results[configuration]["positive_template"] = positive_prompt_template.format(
            reference="[B]", obj1="[A]", relation=relation, obj2="[B]"
        )
    # results[configuration]["negative_template"] = negative_prompt_template.format(
    #     obj1="[A]", relation=relation, obj2="[B]"
    # )
    results[configuration]["x_name"] = x_names[dataset.config["path_type"]]
    results[configuration]["config"] = results[configuration]["data"]["default"][
        "config"
    ]

    with open(results_path, "w") as fp:
        json.dump(results, fp, indent=4)
