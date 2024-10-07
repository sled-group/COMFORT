import torch
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, CLIPImageProcessor
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from llava import conversation as conversation_lib
from glamm import GLaMMForCausalLM, ResizeLongestSide, tokenizer_image_token

from comfort_utils.model_utils.wrapper import VlmWrapper

__all__ = ["GLaMMWrapper"]


def grounding_enc_processor(x: torch.Tensor) -> torch.Tensor:
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    x = (x - IMG_MEAN) / IMG_STD
    h, w = x.shape[-2:]
    x = F.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
    return x



class GLaMMWrapper(VlmWrapper):
    def __init__(self, model_name: str, quantize: bool = False):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.load_model(model_name, quantize)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = ResizeLongestSide(1024)

    def load_model(self, model_name: str, quantize: bool = False):
        kwargs = {"device_map": "auto"}
        if quantize:
            raise ValueError("Quantization is not supported for this model.")
        else:
            # Use torch.float16 for faster inference
            kwargs["torch_dtype"] = torch.float16
        print("Start loading...")
        # init model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.unk_token
        token_idx = {
            "bbox_token_idx": tokenizer("<bbox>", add_special_tokens=False).input_ids[0],
            "seg_token_idx": tokenizer("[SEG]", add_special_tokens=False).input_ids[0],
            "bop_token_idx": tokenizer("<p>", add_special_tokens=False).input_ids[0],
            "eop_token_idx": tokenizer("</p>", add_special_tokens=False).input_ids[0],
        }
        model = GLaMMForCausalLM.from_pretrained(model_name, **kwargs, **token_idx)
        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16)
        self.model = model.bfloat16().cuda()
        self.tokenizer = tokenizer

    def get_logits(
        self,
        images: torch.Tensor | Image.Image,
        prompt: str,
        layer_wise: bool = False,
    ) -> torch.Tensor:
        if layer_wise:
            raise NotImplementedError("Not yet implemented")
        if isinstance(images, Image.Image):
            logits = self.get_logits_single_image(images, prompt)
        elif isinstance(images, torch.Tensor):
            if images.dim() == 3:
                image_pil = ToPILImage()(images)
                logits = self.get_logits_single_image(image_pil, prompt)
            elif images.dim() == 4:
                batch_logits = []
                for image in images:
                    image_pil = ToPILImage()(image)
                    logits = self.get_logits_single_image(image_pil, prompt)
                    batch_logits.append(logits)
                logits = torch.cat(batch_logits, dim=1)
        return logits # vocab_size, batch

    def get_logits_single_image(
        self,
        image: Image.Image,
        prompt: str,
    ) -> torch.Tensor:
        image_np = np.asarray(image)

        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.messages = []
        conv_history = {'user': [], 'model': []}
        conv_history["user"].append(prompt)

        input_str = prompt.replace('&lt;', '<').replace('&gt;', '>')
        prompt = input_str

        DEFAULT_IMAGE_TOKEN = "<image>"
        DEFAULT_IM_START_TOKEN = "<im_start>"
        DEFAULT_IM_END_TOKEN = "<im_end>"

        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = f"The {image_token} provides an overview of the picture." + "\n" + prompt
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        original_size_list = [image_np.shape[:2]]

        global_enc_image = self.global_enc_processor.preprocess(
            image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
        global_enc_image = global_enc_image.bfloat16()
        
        image = self.transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        grounding_enc_image = (grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).
                                                        contiguous()).unsqueeze(0).cuda())
        grounding_enc_image = grounding_enc_image.bfloat16()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        _, _, logits = self.model.evaluate(
            global_enc_image, grounding_enc_image, input_ids, resize_list, original_size_list, max_tokens_new=1,
            bboxes=None)

        logits = logits[0].permute(1, 2, 0)[-1]
        return logits # vocab_size, batch