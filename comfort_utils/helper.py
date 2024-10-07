import torch
from PIL import Image
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.mm_utils import tokenizer_image_token
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os
from .symmetry_metric import normalize_data
from .model_utils import VlmWrapper, GptWrapper
from .model_utils.models_api import query_translation

FOR_MAP_LEFT: dict[str, int] = {
    "camera": { # For English eval, camera is mixed_camera
        "infrontof": "rotated_camera",
        "behind": "rotated_camera",
        "totheleft": "camera",
        "totheright": "camera",
    },
    "mixed_camera": {
        "infrontof": "rotated_camera",
        "behind": "rotated_camera",
        "totheleft": "camera",
        "totheright": "camera",
    },
    "unrotated_camera": {
        "infrontof": "camera",
        "behind": "camera",
        "totheleft": "camera",
        "totheright": "camera",
    },
    "rotated_camera": {
        "infrontof": "rotated_camera",
        "behind": "rotated_camera",
        "totheleft": "rotated_camera",
        "totheright": "rotated_camera",
    },
    "addressee": {
        "infrontof": "rotated_addressee",
        "behind": "rotated_addressee",
        "totheleft": "addressee",
        "totheright": "addressee",
    },
    "mixed_addressee": {
        "infrontof": "rotated_addressee",
        "behind": "rotated_addressee",
        "totheleft": "addressee",
        "totheright": "addressee",
    },
    "unrotated_addressee": {
        "infrontof": "addressee",
        "behind": "addressee",
        "totheleft": "addressee",
        "totheright": "addressee",
    },
    "rotated_addressee": {
        "infrontof": "rotated_addressee",
        "behind": "rotated_addressee",
        "totheleft": "rotated_addressee",
        "totheright": "rotated_addressee",
    },
    "object": {
        "infrontof": "object_facing_left",
        "behind": "object_facing_left",
        "totheleft": "object_facing_left",
        "totheright": "object_facing_left",
    },
    "mixed_object": { # dummy, not existed, ignore
        "infrontof": "object_facing_left",
        "behind": "object_facing_left",
        "totheleft": "object_facing_left",
        "totheright": "object_facing_left",
    },
    "unrotated_object": { # dummy, not existed, ignore
        "infrontof": "object_facing_left",
        "behind": "object_facing_left",
        "totheleft": "object_facing_left",
        "totheright": "object_facing_left",
    },
    "rotated_object": { # dummy, not existed, ignore
        "infrontof": "object_facing_left",
        "behind": "object_facing_left",
        "totheleft": "object_facing_left",
        "totheright": "object_facing_left",
    },
    "nop": { ### FOR PLOTTING in gather_results.py, SHOULD BE REMOVED LATER OR IGNORE
        "infrontof": "rotated_camera",
        "behind": "rotated_camera",
        "totheleft": "camera",
        "totheright": "camera",
    }
} # type: ignore

FOR_MAP_RIGHT: dict[str, int] = {
    "camera": { # For English eval, camera is mixed_camera
        "infrontof": "rotated_camera",
        "behind": "rotated_camera",
        "totheleft": "camera",
        "totheright": "camera",
    },
    "mixed_camera": {
        "infrontof": "rotated_camera",
        "behind": "rotated_camera",
        "totheleft": "camera",
        "totheright": "camera",
    },
    "unrotated_camera": {
        "infrontof": "camera",
        "behind": "camera",
        "totheleft": "camera",
        "totheright": "camera",
    },
    "rotated_camera": {
        "infrontof": "rotated_camera",
        "behind": "rotated_camera",
        "totheleft": "rotated_camera",
        "totheright": "rotated_camera",
    },
    "addressee": {
        "infrontof": "rotated_addressee",
        "behind": "rotated_addressee",
        "totheleft": "addressee",
        "totheright": "addressee",
    },
    "mixed_addressee": {
        "infrontof": "rotated_addressee",
        "behind": "rotated_addressee",
        "totheleft": "addressee",
        "totheright": "addressee",
    },
    "unrotated_addressee": {
        "infrontof": "addressee",
        "behind": "addressee",
        "totheleft": "addressee",
        "totheright": "addressee",
    },
    "rotated_addressee": {
        "infrontof": "rotated_addressee",
        "behind": "rotated_addressee",
        "totheleft": "rotated_addressee",
        "totheright": "rotated_addressee",
    },
    "object": {
        "infrontof": "object_facing_right",
        "behind": "object_facing_right",
        "totheleft": "object_facing_right",
        "totheright": "object_facing_right",
    },
    "mixed_object": { # dummy, not existed, ignore
        "infrontof": "object_facing_right",
        "behind": "object_facing_right",
        "totheleft": "object_facing_right",
        "totheright": "object_facing_right",
    },
    "unrotated_object": { # dummy, not existed, ignore
        "infrontof": "object_facing_right",
        "behind": "object_facing_right",
        "totheleft": "object_facing_right",
        "totheright": "object_facing_right",
    },
    "rotated_object": { # dummy, not existed, ignore
        "infrontof": "object_facing_right",
        "behind": "object_facing_right",
        "totheleft": "object_facing_right",
        "totheright": "object_facing_right",
    },
    "nop": { ### FOR PLOTTING in gather_results.py, SHOULD BE REMOVED LATER OR IGNORE
        "infrontof": "rotated_camera",
        "behind": "rotated_camera",
        "totheleft": "camera",
        "totheright": "camera",
    }
} # type: ignore

FOR_MAP = {"left": FOR_MAP_LEFT, "right": FOR_MAP_RIGHT}

PERSPECTIVE_PROMPT_MAP: dict[str, str] = {
    "nop": "nop",
    "camera1": "camera",
    "camera2": "camera",
    "camera3": "camera",
    "egocentric1": "camera",
    "egocentric2": "camera",
    "egocentric3": "camera",
    "reference1": "object",
    "reference2": "object",
    "reference3": "object",
    "addressee1": "addressee",
    "addressee2": "addressee",
    "addressee3": "addressee",
}

def show_image(image_path):
    with Image.open(image_path) as img:
        plt.imshow(img)
        plt.axis('off')
        plt.show()

def prepare_inputs(image, prompt, image_processor, model, tokenizer):
    """Deprecated function, use prepare_inputs_llava or prepare_inputs_blip instead."""
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = (
        image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        .half()
        .cuda()
    )
    processed_image = Image.fromarray(
        image_tensor[0].permute(1, 2, 0).cpu().numpy().squeeze().astype("uint8")
    )
    inp = f"{roles[0]}: {prompt}"
    inp = (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    )
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(
            raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .cuda()
    )
    """
    [NOTE] To this point, input_ids is in the format of:
        [Instructions], 
        '<', 'im', '_', 'start', '>', 
        '<image>(-200)', 
        '<', 'im', '_', 'end', '>', 
        [Prompt]
    and <image> is one single token (will be replaced by multiple image tokens in the next step)
    """
    (input_ids, position_ids, attention_mask, _, inputs_embeds, _) = (
        model.prepare_inputs_labels_for_multimodal(
            input_ids,
            images=image_tensor,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=None,
        )
    )
    return (
        input_ids,
        inputs_embeds,
        position_ids,
        attention_mask,
        processed_image,
    )


def plot_attention(attention):
    sns.set_theme(style="whitegrid")
    attention = attention.cpu().detach().numpy()
    attention = attention / attention.max()
    plt.figure(figsize=(10, 10))
    sns.heatmap(attention, cmap="YlGnBu")
    plt.show()


def find_index_of_image_tokens(decoded_tokens, image_start_tokens, image_end_tokens):
    for start in range(decoded_tokens.shape[0] - image_start_tokens.shape[0] + 1):
        if torch.all(
            decoded_tokens[start : start + image_start_tokens.shape[0]]
            == image_start_tokens
        ):
            start += image_start_tokens.shape[0]
            break
    for end in range(start, decoded_tokens.shape[0] - image_end_tokens.shape[0] + 1):
        if torch.all(
            decoded_tokens[end : end + image_end_tokens.shape[0]] == image_end_tokens
        ):
            break
    return start, end


def calculate_tokens_mean_attn(attn):
    assert len(attn.shape) == 2, "Attention matrix must be 2D"
    assert attn.shape[0] == attn.shape[1], "Attention matrix must be square"

    # Create a mask for non-zero entries
    non_zero_mask = attn != 0

    # Calculate the sum of attention weights across the third axis, excluding zeros (third axis?)
    sum_weights = torch.where(non_zero_mask, attn, torch.tensor(0.0)).sum(dim=0)  # ????
    # print("sum_weights:", sum_weights[1])
    sum_weights_new = attn.sum(dim=0)
    # print("sum_weights_new:", sum_weights_new[1])

    # Count non-zero entries to use for calculating the mean
    non_zero_count = non_zero_mask.sum(dim=0)
    # print("non_zero_count:", non_zero_count)

    # Calculate the mean by dividing the sum by the number of non-zero entries
    non_zero_means = sum_weights / non_zero_count
    # print("non_zero_means:", non_zero_means)

    return non_zero_means


def prompt_yes_no(
    image,
    prompt,
    image_processor,
    model: LlavaLlamaForCausalLM,
    tokenizer,
):
    """Deprecated function, use prompt_yes_no_new instead."""
    (
        input_ids,
        inputs_embeds,
        position_ids,
        attention_mask,
        _,
    ) = prepare_inputs(image, prompt, image_processor, model, tokenizer)
    # logits = inputs_embeds @ model.get_input_embeddings().weight.T
    # decoded_tokens = logits.argmax(-1)[0]
    # im_start = tokenizer(DEFAULT_IM_START_TOKEN, return_tensors='pt').input_ids[0,1:].to(decoded_tokens.device)
    # im_end = tokenizer(DEFAULT_IM_END_TOKEN, return_tensors='pt').input_ids[0,1:].to(decoded_tokens.device)
    # image_token_start_index, image_token_end_index = find_index_of_image_tokens(decoded_tokens, im_start, im_end)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=True,
        )
    output_logits = output.logits[0, -1, :].cpu().detach().flatten()
    values, indices = torch.topk(torch.softmax(output_logits, dim=-1), 2)
    highest_token_id = indices[0].item()
    second_highest_token_id = indices[1].item()
    highest_token_likelihood = values[0].item()
    second_highest_token_likelihood = values[1].item()
    highest_output_token = tokenizer.decode(highest_token_id).strip()
    second_highest_output_token = tokenizer.decode(second_highest_token_id).strip()
    return (
        highest_output_token,
        highest_token_likelihood,
        second_highest_output_token,
        second_highest_token_likelihood,
    )


def filter_tokens(tokens, filter_set):
    return [token for token in tokens if token not in filter_set]


def prompt_yes_no_new(
    images: torch.Tensor | Image.Image,
    model_wrapper: VlmWrapper,
    prompt: str,
    sum_to_one: bool = True,
    layer_wise: bool = False,
    lang: str = "EN-US",
) -> tuple[str, torch.Tensor, str, torch.Tensor]:
    with torch.inference_mode():
        output_logits = model_wrapper.get_logits(images, prompt, layer_wise)
    if isinstance(model_wrapper, GptWrapper):
        return (
            "Yes",
            output_logits[0],
            "No",
            output_logits[1],
        )
    output_logits = output_logits.to(torch.float32).cpu().detach()
    probs = torch.softmax(output_logits, dim=0)
    yes_set = ["yes", "Yes", "YES", " yes", " Yes", " YES"]
    no_set = ["no", "No", "NO", " no", " No", " NO"]
    translated_yes_set = [query_translation(token, lang) for token in yes_set]
    translated_no_set = [query_translation(token, lang) for token in no_set]
    filter_set = [
        model_wrapper.tokenizer.bos_token_id,
        model_wrapper.tokenizer.eos_token_id,
        model_wrapper.tokenizer.pad_token_id,
        model_wrapper.tokenizer.encode(" ")[-1],
    ]
    yes_ids = set(
        [
            filter_tokens(
                model_wrapper.tokenizer(text=token, return_tensors="pt").input_ids[0],
                filter_set,
            )[0].item()
            for token in translated_yes_set
        ]
    )  # avoid double-counting the same token for different words
    no_ids = set(
        [
            filter_tokens(
                model_wrapper.tokenizer(text=token, return_tensors="pt").input_ids[0],
                filter_set,
            )[0].item()
            for token in translated_no_set
        ]
    )
    yes_prob = probs[torch.tensor(list(yes_ids))].sum(dim=0)
    no_prob = probs[torch.tensor(list(no_ids))].sum(dim=0)
    if sum_to_one:
        total_prob = yes_prob + no_prob
        yes_prob /= total_prob
        no_prob /= total_prob
    return (
        "Yes",
        yes_prob,
        "No",
        no_prob,
    )


def prompt_spatial(
    dataloader,
    model_wrapper: VlmWrapper,
    prompt,
    inner_progress=False,
    desc="",
    sum_to_one=True,
    layer_wise=False,
    lang="EN-US",
):
    stats_prompt = []
    with tqdm(total=len(dataloader), desc=desc, leave=not inner_progress) as data_pbar:
        for batch in dataloader:
            images = batch[0]
            x = batch[1].tolist()
            (
                yes_token,
                highest_token_likelihood,
                no_token,
                second_highest_token_likelihood,
            ) = prompt_yes_no_new(
                images=images,
                model_wrapper=model_wrapper,
                prompt=prompt,
                sum_to_one=sum_to_one,
                layer_wise=layer_wise,
                lang=lang,
            )
            likelihood_dict = [
                {
                    yes_token: highest_token_likelihood[i].tolist(),
                    no_token: second_highest_token_likelihood[i].tolist(),
                }
                for i in range(highest_token_likelihood.shape[0])
            ]
            assert len(x) == len(
                likelihood_dict
            ), f"Length mismatch: x={len(x)}, likelihood_dict={len(likelihood_dict)}"
            stats_prompt.extend(list(zip(x, likelihood_dict)))
            data_pbar.update(1)
    return stats_prompt


def plot_spatial_original(data_for_plot, xlabel, title):
    attribute_data = {}
    for experiment in data_for_plot:
        for point in experiment:
            # point is a tuple of the form (angle, {token1:likelihood1, token2:likelihood2})
            x_value = point[0]
            for token, likelihood in point[1].items():
                if token not in attribute_data:
                    attribute_data[token] = {}
                if x_value not in attribute_data[token]:
                    attribute_data[token][x_value] = []
                attribute_data[token][x_value].append(likelihood)

    # Plotting
    fig, ax = plt.subplots()
    colors = ["blue", "green"]  # Extend this list for more attributes if necessary
    color_index = 0

    for attribute, values in attribute_data.items():
        x = sorted(values.keys())
        y_means = [np.mean(values[x_val]) for x_val in x]
        y_stds = [np.std(values[x_val]) for x_val in x]

        x_float = np.asarray(x, dtype=float)

        ax.plot(x_float, y_means, label=attribute, color=colors[color_index])
        ax.fill_between(
            x_float,
            np.subtract(y_means, y_stds),
            np.add(y_means, y_stds),
            color=colors[color_index],
            alpha=0.2,
        )

        color_index += 1

    ax.set_xlabel(xlabel)
    ax.set_ylabel("prob")
    ax.set_title(title)
    ax.legend()
    plt.savefig(f"plots/spatial_plots/{title}.png")
    # plt.show()


def plot_spatial(
    data_for_plot, xlabel, title, save_path=None, show=False, normalize=False
):
    # if normalize:
    #     data_for_plot = [normalize_data(data) for data in data_for_plot]
    #     title = f"{title} (Normalized)"
    normalized_data_for_plot = [normalize_data(data) for data in data_for_plot]
    attribute_data = {}
    normalized_attribute_data = {}
    gt_curve_data = {}
    x_values = [
        round(point[0][0], 3) if isinstance(point[0], list) else round(point[0], 3)
        for point in data_for_plot[0]
    ]
    for experiment in data_for_plot:
        for i, point in enumerate(experiment):
            # point is a tuple of the form (angle, {token1:likelihood1, token2:likelihood2})
            x_value = x_values[i]
            for token, likelihood in point[1].items():
                if token != "No":
                    if token not in attribute_data:
                        attribute_data[token] = {}
                        gt_curve_data[token] = {}
                    if x_value not in attribute_data[token]:
                        attribute_data[token][x_value] = []
                        gt_curve_data[token][x_value] = []
                    attribute_data[token][x_value].append(likelihood)
                    gt = (np.cos((x_value - 180) / 180 * np.pi) + 1) / 2
                    gt_curve_data[token][x_value].append(gt)
    for experiment in normalized_data_for_plot:
        for i, point in enumerate(experiment):
            # point is a tuple of the form (angle, {token1:likelihood1, token2:likelihood2})
            x_value = x_values[i]
            for token, likelihood in point[1].items():
                if token != "No":
                    if token not in normalized_attribute_data:
                        normalized_attribute_data[token] = {}
                    if x_value not in normalized_attribute_data[token]:
                        normalized_attribute_data[token][x_value] = []
                    normalized_attribute_data[token][x_value].append(likelihood)

    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 15))
    plt.tight_layout(pad=1.0)
    sns.set_context("notebook", font_scale=1.5)
    _, ax = plt.subplots()
    colors = ["red", "blue", "green"]  # Extend this list for more attributes if necessary
    color_index = 0

    for data_i, this_attribute_data in enumerate([attribute_data, normalized_attribute_data, gt_curve_data]):
        for attribute, values in this_attribute_data.items():
            x = sorted(values.keys())
            y_means = [np.mean(values[x_val]) for x_val in x]
            y_stds = [np.std(values[x_val]) for x_val in x]
            x_float = np.asarray(x, dtype=float)

            if data_i == 0:
                # ax.plot(x_float, y_means, label=f"P('{attribute}')", color=colors[color_index])
                sns.lineplot(x=x_float, y=y_means, label=f"P('{attribute}')", color=colors[data_i % len(colors)])
            elif data_i == 1:
                # ax.plot(x_float, y_means, label=f"Normalized P('{attribute}')", color=colors[color_index])
                sns.lineplot(x=x_float, y=y_means, label=f"Normalized P('{attribute}')", color=colors[data_i % len(colors)])
            elif data_i == 2:
                # ax.plot(x_float, y_means, label=f"Ground Truth P('{attribute}')", color=colors[color_index])
                sns.lineplot(x=x_float, y=y_means, label=f"Ground Truth P('{attribute}')", color=colors[data_i % len(colors)])
            plt.fill_between(x_float, np.subtract(y_means, y_stds), np.add(y_means, y_stds), color=colors[data_i % len(colors)], alpha=0.2)
            # ax.fill_between(
            #     x_float,
            #     np.subtract(y_means, y_stds),
            #     np.add(y_means, y_stds),
            #     color=colors[color_index],
            #     alpha=0.2,
            # )

            color_index += 1

    # x = np.linspace(0, 360, 37)
    # ax.plot(x, (np.cos((x - 180) / 180 * np.pi) + 1) / 2, 'r', label='cos(θ)')
    ax.set_xlim([0, 360])
    ax.set_ylim([0, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("prob")
    # ax.set_title(title)
    ax.legend()
    # plt.tight_layout()
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"{title}.png"), bbox_inches="tight")
        plt.close()
    if show:
        plt.show()


def get_prompt_template(perspective_prompt):
    if perspective_prompt == "nop":
        positive_prompt_template = "Is the {obj1} {relation} the {obj2}?"
    elif perspective_prompt == "camera1":
        positive_prompt_template = (
            "From the camera's perspective, is the {obj1} {relation} the {obj2}?"
        )
    elif perspective_prompt == "camera2":
        positive_prompt_template = (
            "From the camera's frame of reference, is the {obj1} {relation} the {obj2}?"
        )
    elif perspective_prompt == "camera3":
        positive_prompt_template = (
            "From the camera's viewpoint, is the {obj1} {relation} the {obj2}?"
        )
    elif perspective_prompt == "egocentric3":
        positive_prompt_template = (
            "From the egocentric viewpoint, is the {obj1} {relation} the {obj2}?"
        )
    elif perspective_prompt == "addressee1":
        positive_prompt_template = (
            "From the woman's perspective, is the {obj1} {relation} the {obj2}?"
        )
    elif perspective_prompt == "addressee2":
        positive_prompt_template = (
            "From the woman's frame of reference, is the {obj1} {relation} the {obj2}?"
        )
    elif perspective_prompt == "addressee3":
        positive_prompt_template = (
            "From the woman's viewpoint, is the {obj1} {relation} the {obj2}?"
        )
    elif perspective_prompt == "reference1":
        positive_prompt_template = (
            "From the {reference}'s perspective, is the {obj1} {relation} the {obj2}?"
        )
    elif perspective_prompt == "reference2":
        positive_prompt_template = (
            "From the {reference}'s frame of reference, is the {obj1} {relation} the {obj2}?"
        )
    elif perspective_prompt == "reference3":
        positive_prompt_template = (
            "From the {reference}'s viewpoint, is the {obj1} {relation} the {obj2}?"
        )
    else:
        raise Exception(f"Invalid perspective prompt: {perspective_prompt}")
    return positive_prompt_template
