# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from torchvision import transforms

from comfort_utils.model_utils.wrapper import VlmWrapper

__all__ = ["MiniCpmWrapper"]

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_MEAN
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD


class MiniCpmWrapper(VlmWrapper):
    def __init__(self, model_name: str, quantize: bool = False):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.load_model(model_name, quantize)
        self.transforms = transforms.Normalize(
            mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
        )

    def load_model(self, model_name: str, quantize: bool = False):
        kwargs = {}
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, **kwargs, trust_remote_code=True
        ).eval().cuda()
        self.image_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("Finished loading...")

    def get_logits(
        self,
        images: torch.Tensor | Image.Image,
        prompt: str,
        layer_wise: bool = False,
    ) -> torch.Tensor:
        if layer_wise:
            raise NotImplementedError("Not yet implemented")
        images = self.transforms(images)
        image_token = "(<image>./</image>)"
        msgs = [{"role": "user", "content": image_token + '\n' + prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            batch_size = images.shape[0]
        else:
            batch_size = 1
        if isinstance(images, Image.Image):
            raise NotImplementedError("Not yet implemented for Image.Image input type.")
        elif isinstance(images, torch.Tensor):
            if images.ndim == 3:
                img_list = [images.to(self.model.dtype)]
            elif images.ndim == 4:
                images = images.to(self.model.dtype)
                img_list = [image for image in images]
            else:
                raise ValueError("Invalid input shape")

        logits = []
        for img in img_list:
            model_inputs = self.image_processor(chat_prompt, [img], return_tensors="pt")
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    model_inputs[k] = v.cuda()
            pixel_values_list = model_inputs['pixel_values']
            for i in range(len(pixel_values_list)):
                for j in range(len(pixel_values_list[i])):
                    pixel_values_list[i][j] = pixel_values_list[i][j].cuda()
            model_inputs['pixel_values'] = pixel_values_list
            vllm_embedding, _ = self.model.get_vllm_embedding(model_inputs)
            output = self.model.llm(
                input_ids=None,
                inputs_embeds=vllm_embedding,
            )
            logit = output.logits.permute(1, 2, 0)[-1]
            logits.append(logit)
        logits = torch.cat(logits, dim=1)
        return logits
