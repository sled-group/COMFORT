from openai import OpenAI
from torch import Tensor
from PIL.Image import Image
from .wrapper import VlmWrapper
from .api_keys import APIKEY_OPENAI
from .models_api import encode_image
from torchvision.transforms import ToPILImage
import numpy as np
import requests
import torch

SYSTEM_PROMPT = 'You will be provided an image and a question, please answer the question only in "Yes" or "No"'

def sub_string_tokens_in_resp_token(tokens, resp_token):
    for token in tokens:
        if token in resp_token:
            return True
    return False

class GptWrapper(VlmWrapper):
    def __init__(self, model_name: str):
        self.load_model(model_name)
        
    def load_model(self, model_name: str, quantize: bool = False):
        self.model = model_name
        self.client = OpenAI(api_key=APIKEY_OPENAI)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {APIKEY_OPENAI}"
        }

    def prepare_inputs(
        self,
        image: Tensor | Image,
        prompt: str,
    ):
        """Intended for single image input only"""
        if isinstance(image, Tensor):
            assert image.dim() == 3, "Input image should be 3D tensor"
            image = ToPILImage()(image)
        base64_image = encode_image(image)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]
        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "logprobs": True,
            "top_logprobs": 4,
            "max_tokens": 300
        }
        return payload
        # return messages

    def get_logits(
        self,
        images: Tensor | Image,
        prompt: str,
        layer_wise: bool = False,
    ) -> Tensor:
        if layer_wise:
            raise ValueError("Layer-wise logit retrieval is not supported")
        if isinstance(images, Tensor) and images.dim() == 4:
            probs = []
            for image in images:
                probs.append(self.get_logits_single_image(image, prompt))
            probs = torch.stack(probs, dim=-1)
            # probs: (2, num_images)
            return probs
        else:
            return self.get_logits_single_image(images, prompt)
    
    def get_logits_single_image(
        self,
        image: Tensor | Image,
        prompt: str,
    ) -> Tensor:
        """Returns the probabilities of Yes and No in a Tensor of shape (2,)"""
        payload = self.prepare_inputs(image, prompt)
        first_trial = True
        while first_trial == True or response.status_code != 200:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
            first_trial = False
            # print("response.status_code:", response.status_code)
            if response.status_code != 200:
                print(f"Erroneous Status Code: {response.status_code}. Retrying...")
        # print("goes to here")
        response = response.json()
        logprobs = response["choices"][0]["logprobs"]
        top_logprobs = logprobs["content"][0]["top_logprobs"]
        yes_tokens = ["Yes", "yes", "YES"]
        no_tokens = ["No", "no", "NO"]
        # print(top_logprobs)
        yes_logprobs = []
        no_logprobs = []
        for logprob in top_logprobs:
            token = logprob["token"]
            token = token.replace("_", "").strip()
            if token in yes_tokens: # or sub_string_tokens_in_resp_token(yes_tokens, token):
                # print("Yes token:", token)
                yes_logprobs.append(logprob["logprob"])
            elif token in no_tokens: # or sub_string_tokens_in_resp_token(no_tokens, token):
                # print("No token:", token)
                no_logprobs.append(logprob["logprob"])
            else:
                pass
                # print("Unknown token:", token)
        # print("Yes logprobs:", yes_logprobs)
        # print("No logprobs:", no_logprobs)
        yes_prob = np.exp(yes_logprobs).sum()
        no_prob = np.exp(no_logprobs).sum()
        # print("Yes prob:", yes_prob)
        # print("No prob:", no_prob)
        probs = torch.tensor([yes_prob, no_prob])
        probs = probs / probs.sum()
        # print(probs)
        return probs
