from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from comfort_utils.model_utils.wrapper import VlmWrapper

__all__ = ["XComposerWrapper"]


class XComposerWrapper(VlmWrapper):
    def __init__(self, model_name: str, quantize: bool = False):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.load_model(model_name, quantize)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.model.config.img_size, self.model.config.img_size),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def load_model(self, model_name: str, quantize: bool = False):
        kwargs = {"device_map": "auto"}
        if quantize:
            raise ValueError("Quantization is not supported for this model.")
        else:
            # Use torch.float16 for faster inference
            kwargs["torch_dtype"] = torch.float16
        print("Start loading...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, **kwargs, trust_remote_code=True
        ).eval()
        if hasattr(self.model, "tokenizer") and self.model.tokenizer is None:
            self.model.tokenizer = self.tokenizer
        print("Finished loading...")

    def get_logits(
        self,
        images: torch.Tensor | Image.Image,
        prompt: str,
        layer_wise: bool = False,
    ) -> torch.Tensor:
        if layer_wise:
            raise NotImplementedError("Not yet implemented")
        prompt = f"""[UNUSED_TOKEN_146]user\n<ImageHere>{prompt}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
        num_images = 0
        if isinstance(images, Image.Image):
            images = [images]
            num_images = 1
        elif isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = [self.transform(images).unsqueeze(0).to(self.model.dtype)]
                num_images = 1
            elif images.dim() == 4:
                images = [
                    self.transform(image).unsqueeze(0).to(self.model.dtype)
                    for image in images
                ]
                num_images = len(images)
        else:
            raise ValueError("Invalid input type for images.")
        samples = {
            "data_type": ["multi"],
            "text_input": [[prompt] for _ in range(num_images)],
            "image": images,
        }
        output = self.model.forward(samples=samples)
        logits = output.logits.permute(1, 2, 0)[-1]
        return logits
