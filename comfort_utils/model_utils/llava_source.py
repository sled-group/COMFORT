import torch
from PIL import Image
from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.mm_utils import tokenizer_image_token

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
)

from comfort_utils.model_utils.wrapper import VlmWrapper

__all__ = ["LlavaWrapper"]


def load_llava(
    model_name: str, quantize: bool = False
) -> tuple[LlavaLlamaForCausalLM, AutoTokenizer, CLIPImageProcessor]:
    if model_name in [
        "liuhaotian/llava-v1.5-7b",
        "liuhaotian/llava-v1.5-13b",
        "remyxai/SpaceLLaVA"
    ]:
        model_path = model_name
        kwargs = {"device_map": "auto"}
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
        print("Start loading...")
        model = LlavaLlamaForCausalLM.from_pretrained(model_path, **kwargs)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        print("Loading vision processor...")
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device="cuda")
        image_processor = vision_tower.image_processor
        print("Finished loading...")
        return model, tokenizer, image_processor
    else:
        raise ValueError(f"Model {model_name} not found")


def prepare_inputs_llava(
    image: torch.Tensor | Image.Image,
    prompt: str,
    image_processor: CLIPImageProcessor,
    tokenizer: AutoTokenizer,
):
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = (
        image_processor.preprocess(image, return_tensors="pt", do_rescale=False)[
            "pixel_values"
        ]
        .half()
        .cuda()
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
    return (
        input_ids,
        image_tensor,
    )


def get_logits_llava(
    model: LlavaLlamaForCausalLM,
    images: torch.Tensor | Image.Image,
    prompt: str,
    image_processor: CLIPImageProcessor,
    tokenizer: AutoTokenizer,
    layer_wise: bool = False,
):
    (
        input_ids,
        image_tensor,
    ) = prepare_inputs_llava(images, prompt, image_processor, tokenizer)
    if input_ids.shape[0] == 1 and image_tensor.shape[0] != 1:
        input_ids = input_ids.repeat(image_tensor.shape[0], 1)
    if layer_wise:
        output = model.forward(
            input_ids=input_ids,
            images=image_tensor,
            output_hidden_states=True,
        )
        hidden_states = torch.stack(output.hidden_states, dim=0)
        output_logits = model.lm_head(hidden_states)
        output_logits = output_logits.permute(2, 3, 1, 0)[
            -1
        ]  # vocab_size, batch, num_layers
    else:
        output = model.forward(
            input_ids=input_ids,
            images=image_tensor,
        )
        # print(output.logits.shape) # batch, num_tokens, vocab_size
        output_logits = output.logits.permute(1, 2, 0)[-1]  # vocab_size, batch
    return output_logits


class LlavaWrapper(VlmWrapper):
    def __init__(self, model_name: str, quantize: bool = False):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.load_model(model_name, quantize)

    def load_model(self, model_name: str, quantize: bool = False):
        self.model, self.tokenizer, self.image_processor = load_llava(
            model_name, quantize
        )

    def prepare_inputs(
        self,
        image: torch.Tensor | Image.Image,
        prompt: str,
    ):
        return prepare_inputs_llava(image, prompt, self.image_processor, self.tokenizer)

    def get_logits(
        self,
        images: torch.Tensor | Image.Image,
        prompt: str,
        layer_wise: bool = False,
    ) -> torch.Tensor:
        return get_logits_llava(
            self.model, images, prompt, self.image_processor, self.tokenizer, layer_wise
        )
