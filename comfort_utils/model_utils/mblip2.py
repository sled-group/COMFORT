import torch
from PIL import Image
from comfort_utils.model_utils.wrapper import VlmWrapper
from comfort_utils.model_utils.blip2 import BlipWrapper


class MBlipWrapper(BlipWrapper):
    def __init__(self, model_name: str, quantize: bool = False):
        super().__init__(model_name, quantize)
        self.instruction = ""

    def set_instruction(self, instruction: str):
        self.instruction = instruction

    def prepare_inputs(
        self,
        image: torch.Tensor | Image.Image,
        prompts: list[str],
    ):
        prompts = [self.instruction + '\n' + p for p in prompts]
        inputs = self.image_processor(images=image, text=prompts, return_tensors="pt").to("cuda")
        return inputs
