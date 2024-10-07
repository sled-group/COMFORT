import torch
from PIL import Image

from transformers import (
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)

from comfort_utils.model_utils.wrapper import VlmWrapper

__all__ = ["LlavaHfWrapper"]


def load_llava_hf(model_name: str, quantize: bool = False):
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
    processor = LlavaProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(model_name, **kwargs)
    model.eval()
    print("Finished loading...")
    return model, processor, None


def prepare_inputs_llava_hf(
    image: torch.Tensor | Image.Image,
    prompt: str,
    processor: LlavaProcessor,
):
    prompt = (
        f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if isinstance(image, torch.Tensor) and image.ndim == 4:
        print(image.shape)
        prompt = [prompt] * image.shape[0]
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        "cuda", torch.float16
    )
    return inputs


def get_logits_llava_hf(
    model: LlavaForConditionalGeneration,
    images: torch.Tensor | Image.Image,
    prompt: str,
    image_processor: None,
    tokenizer: LlavaProcessor,
    layer_wise: bool = False,
):
    inputs = prepare_inputs_llava_hf(images, [prompt] * len(images), tokenizer)
    if layer_wise:
        output = model.forward(**inputs, output_hidden_states=True)
        hidden_states = torch.stack(output.language_model_outputs.hidden_states, dim=0)
        output_logits = model.language_model.lm_head(hidden_states)
        output_logits = output_logits.permute(2, 3, 1, 0)[
            -1
        ]  # vocab_size, batch, num_layers
    else:
        output = model.forward(**inputs)
        # print(output.logits.shape) # batch, num_tokens, vocab_size
        output_logits = output.logits.permute(1, 2, 0)[-1]  # vocab_size, batch
    return output_logits


class LlavaHfWrapper(VlmWrapper):
    def __init__(self, model_name: str, quantize: bool = False):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.load_model(model_name, quantize)

    def load_model(self, model_name: str, quantize: bool = False):
        self.model, self.tokenizer, self.image_processor = load_llava_hf(
            model_name, quantize
        )

    def prepare_inputs(
        self,
        image: torch.Tensor | Image.Image,
        prompt: str,
    ):
        return prepare_inputs_llava_hf(
            image, prompt, self.image_processor, self.tokenizer
        )

    def get_logits(
        self,
        images: torch.Tensor | Image.Image,
        prompt: str,
        layer_wise: bool = False,
    ) -> torch.Tensor:
        return get_logits_llava_hf(
            self.model, images, prompt, self.image_processor, self.tokenizer, layer_wise
        )
