import sys
import cv2
import random
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.generate_utils import center_crop, create_feathered_mask
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from tools.markdown_utils import (markdown_default, examples, title, description, article, process_markdown, colors,
                                  draw_bbox, ImageSketcher)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

def parse_args(args):
    parser = argparse.ArgumentParser(description="GLaMM Model Demo")
    parser.add_argument("--version", default="MBZUAI/GLaMM-FullScope")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--precision", default='bf16', type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    return parser.parse_args(args)

def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token
    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    return tokenizer

def initialize_model(args, tokenizer):
    """ Initialize the GLaMM model. """
    model_args = {k: getattr(args, k) for k in
                  ["seg_token_idx", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}

    model = GLaMMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args)
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model

def prepare_model_for_inference(model, args):
    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)
    model = model.bfloat16().cuda()
    return model

def grounding_enc_processor(x: torch.Tensor) -> torch.Tensor:
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    x = (x - IMG_MEAN) / IMG_STD
    h, w = x.shape[-2:]
    x = F.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
    return x

def region_enc_processor(orig_size, post_size, bbox_img):
    orig_h, orig_w = orig_size
    post_h, post_w = post_size
    y_scale = post_h / orig_h
    x_scale = post_w / orig_w
    print("orig_h",orig_h)
    print("orig_w",orig_w)
    bboxes_scaled = [[bbox[0] * x_scale, bbox[1] * y_scale, bbox[2] * x_scale, bbox[3] * y_scale] for bbox in bbox_img]

    tensor_list = []
    for box_element in bboxes_scaled:
        ori_bboxes = np.array([box_element], dtype=np.float64)
        # Normalizing the bounding boxes
        norm_bboxes = ori_bboxes / np.array([post_w, post_h, post_w, post_h])
        # Converting to tensor, handling device and data type as in the original code
        tensor_list.append(torch.tensor(norm_bboxes, device='cuda').half().to(torch.bfloat16))

    if len(tensor_list) > 1:
        bboxes = torch.stack(tensor_list, dim=1)
        bboxes = [bboxes.squeeze()]
    else:
        bboxes = tensor_list
    return bboxes

def prepare_mask(input_image, image_np, pred_masks, text_output, color_history):
    save_img = None
    for i, pred_mask in enumerate(pred_masks):
        if (pred_mask.shape[0] == 0):
            continue
        pred_mask = pred_mask.detach().cpu().numpy()
        mask_list = [pred_mask[i] for i in range(pred_mask.shape[0])]
        if len(mask_list) > 0:
            save_img = image_np.copy()
            colors_temp = colors
            seg_count = text_output.count("[SEG]")
            mask_list = mask_list[-seg_count:]
            for curr_mask in mask_list:
                color = random.choice(colors_temp)
                if len(colors_temp) > 0:
                    colors_temp.remove(color)
                else:
                    colors_temp = colors
                color_history.append(color)
                curr_mask = curr_mask > 0
                save_img[curr_mask] = (image_np * 0.5 + curr_mask[:, :, None].astype(np.uint8) * np.array(color) * 0.5)[
                    curr_mask]
    seg_mask = np.zeros((curr_mask.shape[0], curr_mask.shape[1], 3), dtype=np.uint8)
    seg_mask[curr_mask] = [255, 255, 255]  # white for True values
    seg_mask[~curr_mask] = [0, 0, 0]  # black for False values
    seg_mask = Image.fromarray(seg_mask)
    # mask_path = input_image.replace('image', 'mask')
    # seg_mask.save(mask_path)

    return save_img

def generate_new_image(st_pipe, input_str, input_image, mask_path):
    if mask_path is None:
        raise ValueError("No Segmentation Mask")

    og_image = load_image(input_image)
    st_image, c_box = center_crop(og_image)
    im_height = st_image.size[0]
    st_image = st_image.resize((1024, 1024))
    st_mask = load_image(mask_path)
    st_mask, c_box = center_crop(st_mask)
    st_mask = st_mask.resize((1024, 1024))

    st_generator = torch.Generator(device="cuda").manual_seed(0)
    st_out = st_pipe(
        prompt=input_str, image=st_image, mask_image=st_mask, guidance_scale=8.0, num_inference_steps=20, strength=0.99,
        generator=st_generator, ).images[0]

    st_out = st_out.resize((im_height, im_height))
    feathered_mask = create_feathered_mask(st_out.size)
    og_image.paste(st_out, c_box, feathered_mask)
    st_text_out = "Sure, Here's the new image"
    st_text_out = process_markdown(st_text_out, [])

    return og_image, st_text_out

def inference(input_str, input_image, bbox_img, follow_up=False, generate=False):
    print("input_str: ", input_str, "input_image: ", input_image)

    if generate:
        return generate_new_image(st_pipe, input_str, input_image, mask_path)

    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    conv_history = {'user': [], 'model': []}
    conv_history["user"].append(input_str)

    input_str = input_str.replace('&lt;', '<').replace('&gt;', '>')
    prompt = input_str
    prompt = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture." + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.imread(input_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_np.shape[:2]
    original_size_list = [image_np.shape[:2]]

    global_enc_image = global_enc_processor.preprocess(
        image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
    global_enc_image = global_enc_image.bfloat16()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    grounding_enc_image = (grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).
                                                   contiguous()).unsqueeze(0).cuda())
    grounding_enc_image = grounding_enc_image.bfloat16()

    post_h, post_w = global_enc_image.shape[1:3]
    bboxes = None
    if len(bbox_img) > 0:
        bboxes = region_enc_processor((orig_h, orig_w), (post_h, post_w), bbox_img)

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks, logits = model.evaluate(
        global_enc_image, grounding_enc_image, input_ids, resize_list, original_size_list, max_tokens_new=512,
        bboxes=bboxes)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    print("LOGITS:",logits)
    logits_shape = len(logits)
    logits_tensor = logits[0]
    logits_tensor = logits_tensor.to(torch.float32)  # 将bfloat16转换为float32
    print("shape of logits_tensor",logits_tensor.shape )
    probs = F.softmax(logits_tensor, dim=-1)
    print("shape of probs",probs.shape )
    # initial_probs = probs[0, 0, :]

    # # 获取每个位置的概率最高的10个索引及其对应的概率
    # top_probs, top_indices = torch.topk(initial_probs, 50, dim=-1)

    # # 打印每个位置概率最高的10个单词及其概率
    # print(f"Initial word probabilities:")
    # for j in range(50):
    #     word = tokenizer.decode([top_indices[j].item()])
    #     prob = top_probs[j].item()
    #     print(f"  {word}: {prob:.4f}")
    initial_probs = probs[0, -1, :]  # 获取第一个位置的概率分布

    # 类别列表
    categories = [
        "dining table", "cow", "spoon", "potted plant", "zebra", "donut", "traffic light", "backpack", "vase", 
        "sports ball", "person", "apple", "chair", "bicycle", "pizza", "remote", "tennis racket", "bench", "boat", 
        "motorcycle", "bottle", "broccoli", "suitcase", "dog", "orange", "skis", "cake", "umbrella", "cell phone", 
        "bowl", "wine glass", "surfboard", "handbag", "sheep", "tie", "banana", "sink", "kite", "clock", "cup", 
        "giraffe", "bus", "knife", "bird", "elephant", "truck", "book", "car", "tv", "carrot"
    ]

    # 获取每个类别在词汇表中的索引
    category_indices = [tokenizer.encode(category, add_special_tokens=False)[0] for category in categories]

    # 打印每个类别对应的概率
    print(f"Probabilities for specified categories at initial position:")
    category_probs = []
    for category, index in zip(categories, category_indices):
        prob = initial_probs[index].item()
        category_probs.append((category, prob))
        print(f"  {category}: {prob:.4e}")
    max_category, max_prob = max(category_probs, key=lambda x: x[1])
    print(f"\nCategory with the highest probability:")
    print(f"  {max_category}: {max_prob:.4e}")
    print("logits shape: ",logits_shape)
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]
    print("text_output: ", text_output)

    conv.messages.pop()
    conv.append_message(conv.roles[1], text_output)
    conv_history["model"].append(text_output)
    color_history = []
    save_img = None
    if "[SEG]" in text_output:
        save_img = prepare_mask(input_image, image_np, pred_masks, text_output, color_history)

    output_str = text_output
    if save_img is not None:
        output_image = save_img
    else:
        if len(bbox_img) > 0:
            output_image = draw_bbox(image_np.copy(), bbox_img)
        else:
            output_image = input_image

    markdown_out = process_markdown(output_str, color_history)

    return output_image, markdown_out

# Function to get word probabilities
def get_word_probabilities(model, tokenizer, input_str):
    # Tokenize the input string
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids.cuda()
    
    # Get the logits from the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        print("logits:", outputs.logits)
        logits = outputs.logits

    # Get the probabilities for the last token in the input
    probabilities = F.softmax(logits[0, -1], dim=-1).cpu().numpy()
    
    # Get the vocabulary and map the probabilities to words
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    word_probs = {inv_vocab[i]: prob for i, prob in enumerate(probabilities)}
    
    return word_probs

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    model = prepare_model_for_inference(model, args)
    global_enc_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    st_pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

    conv = None
    conv_history = {'user': [], 'model': []}
    mask_path = None

    # Example input
    input_str = """Select the class for <bbox> from: dining table, cow, spoon, potted plant, zebra, donut, traffic light, backpack, vase, sports ball, person, apple, chair, bicycle, pizza, remote, tennis racket, bench, boat, motorcycle, bottle, broccoli, suitcase, dog, orange, skis, cake, umbrella, cell phone, bowl, wine glass, surfboard, handbag, sheep, tie, banana, sink, kite, clock, cup, giraffe, bus, knife, bird, elephant, truck, book, car, tv, carrot."""
    input_image = "/nfs/turbo/coe-chaijy/xuejunzh/data/validation/image/mixed-images/raw/ADE/ADE_val_00000285.jpg"
    bbox_img = [[497,146,590,700]]  # Example bbox coordinates
    # tokens = tokenizer.tokenize(input_str)
    # # Print out the number of tokens
    # print("Number of tokens:", len(tokens))
    # # Optionally, you can also print the tokens themselves to inspect
    # print("Tokens:", tokens)
    output_image, output_text = inference(input_str, input_image, bbox_img)
    # output_image.save("output_image.jpg")
    # print("Output Text:", output_text)

    # # Get word probabilities
    # word_probs = get_word_probabilities(model, tokenizer, input_str)
    # # Print probabilities for top 10 words
    # for word, prob in sorted(word_probs.items(), key=lambda item: item[1], reverse=True)[:10]:
    #     print(f"{word}: {prob:.4f}")
