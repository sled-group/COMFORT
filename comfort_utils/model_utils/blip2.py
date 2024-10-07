import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
)

from comfort_utils.model_utils.wrapper import VlmWrapper

BLIP2_PROMPT_TEMPLATE = "Question: {question}\nAnswer: "


class BlipWrapper(VlmWrapper):
    def __init__(self, model_name: str, quantize: bool = False):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.load_model(model_name, quantize)

    def load_model(self, model_name: str, quantize: bool = False):
        if "flan" in model_name:
            raise NotImplementedError("FLAN models are not supported, use opt instead.")
        kwargs = {}
        if quantize:
            print("Loading a quantized model...")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        else:
            # Use torch.float16 for faster inference
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto"
        print("Start loading...")
        if "instruct" in model_name:
            processor = InstructBlipProcessor.from_pretrained(model_name)
            model = InstructBlipForConditionalGeneration.from_pretrained(
                model_name, **kwargs
            )
        else:
            processor = Blip2Processor.from_pretrained(model_name)
            model = Blip2ForConditionalGeneration.from_pretrained(model_name, **kwargs)
        model.eval()
        # if not quantize:
        #     # Model is on cpu since device_map is not set
        #     model.to("cuda")
        print("Finished loading...")
        self.model = model
        self.image_processor = processor
        self.tokenizer = self.image_processor.tokenizer

    def prepare_inputs(
        self,
        image: torch.Tensor | Image.Image,
        prompts: list[str],
    ):
        prompt = [BLIP2_PROMPT_TEMPLATE.format(question=p) for p in prompts]
        inputs = self.image_processor(images=image, text=prompt, return_tensors="pt").to("cuda")
        return inputs

    def get_logits(
        self, images: torch.Tensor | Image.Image, prompt: str, layer_wise: bool = False
    ) -> torch.Tensor:
        num_images = -1
        if isinstance(images, Image.Image):
            num_images = 1
        elif isinstance(images, torch.Tensor):
            if len(images.shape) == 4:
                num_images = images.shape[0]
            elif len(images.shape) == 3:
                num_images = 1
        if num_images == -1:
            raise ValueError("Invalid input shape")
        inputs = self.prepare_inputs(images, [prompt] * num_images)
        if layer_wise:
            output = self.model.forward(**inputs, output_hidden_states=True)
            hidden_states = torch.stack(output.language_model_outputs.hidden_states, dim=0)
            output_logits = self.model.language_model.lm_head(hidden_states)
            output_logits = output_logits.permute(2, 3, 1, 0)[
                -1
            ]  # vocab_size, batch, num_layers
        else:
            output = self.model.forward(**inputs)
            # print(output.logits.shape) # batch, num_tokens, vocab_size
            output_logits = output.logits.permute(1, 2, 0)[-1]  # vocab_size, batch
        return output_logits
