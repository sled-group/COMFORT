from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from internvl.conversation import get_conv_template

from comfort_utils.model_utils.wrapper import VlmWrapper


__all__ = ["InternVlWrapper"]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def process_image(image: torch.Tensor, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVlWrapper(VlmWrapper):
    def __init__(self, model_name: str, quantize: bool = False):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.load_model(model_name, quantize)

    def load_model(self, model_name: str, quantize: bool = False):
        kwargs = {"device_map": "auto", "trust_remote_code": True}
        if quantize:
            print("Loading a quantized model...")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            # Use torch.float16 for faster inference
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModel.from_pretrained(model_name, **kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    def get_logits(
        self, images: torch.Tensor | Image.Image, prompt: str, layer_wise: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError("This model is not fully implemented yet.")
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            pixel_values = [process_image(img) for img in images]
            pixel_values = torch.stack(pixel_values)
        else:
            pixel_values = process_image(images)

        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        template = get_conv_template(self.template)

        image_bs = pixel_values.shape[0]
        print(f"dynamic ViT batch size: {image_bs}")
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * self.num_image_token * image_bs
            + IMG_END_TOKEN
        )
        question = image_tokens + "\n" + prompt
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].cuda()
        attention_mask = model_inputs["attention_mask"].cuda()

        if layer_wise:
            output = self.model.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = torch.stack(
                output.language_model_outputs.hidden_states, dim=0
            )
            output_logits = self.model.language_model.lm_head(hidden_states)
            output_logits = output_logits.permute(2, 3, 1, 0)[
                -1
            ]  # vocab_size, batch, num_layers
        else:
            output = self.model.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # print(output.logits.shape) # batch, num_tokens, vocab_size
            output_logits = output.logits.permute(1, 2, 0)[-1]  # vocab_size, batch

        return output_logits


# tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
# # set the max number of tiles in `max_num`
# pixel_values = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()

# generation_config = dict(
#     num_beams=1,
#     max_new_tokens=512,
#     do_sample=False,
# )

# # single-round single-image conversation
# question = "请详细描述图片" # Please describe the picture in detail
# response = model.chat(tokenizer, pixel_values, question, generation_config)
# print(question, response)

# # multi-round single-image conversation
# question = "请详细描述图片" # Please describe the picture in detail
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
# print(question, response)

# question = "请根据图片写一首诗" # Please write a poem according to the picture
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(question, response)

# # multi-round multi-image conversation
# pixel_values1 = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=6).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# question = "详细描述这两张图片" # Describe the two pictures in detail
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
# print(question, response)

# question = "这两张图片的相同点和区别分别是什么" # What are the similarities and differences between these two pictures
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(question, response)

# # batch inference (single image per sample)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=6).to(torch.bfloat16).cuda()
# image_counts = [pixel_values1.size(0), pixel_values2.size(0)]
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# questions = ["Describe the image in detail."] * len(image_counts)
# responses = model.batch_chat(tokenizer, pixel_values,
#                              image_counts=image_counts,
#                              questions=questions,
#                              generation_config=generation_config)
# for question, response in zip(questions, responses):
#     print(question)
#     print(response)
