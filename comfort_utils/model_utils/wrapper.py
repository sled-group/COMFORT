from abc import ABC, abstractmethod
from PIL.Image import Image
from torch import Tensor


class VlmWrapper(ABC):
    model: object
    image_processor: object
    tokenizer: object

    @abstractmethod
    def load_model(self, model_name: str, quantize: bool = False):
        """Load the model.
        
        Args:
            model_name (str): Model name.
            quantize (bool, optional): Whether to 4-bit quantize the model. Defaults to False.
        """
        pass

    @abstractmethod
    def prepare_inputs(
        self,
        image: Tensor | Image,
        prompt: str,
    ):
        """Prepare inputs for the model.
        
        Args:
            image (Tensor | Image): Image tensor or PIL Image.
            prompt (str): Prompt string.

        Returns:
            dict: Dictionary containing the inputs for the model, so that self.model.forward(**inputs) can be called.
        """
        pass

    @abstractmethod
    def get_logits(
        self,
        images: Tensor | Image,
        prompt: str,
        layer_wise: bool = False,
    ) -> Tensor:
        """Get logits from the model.
        
        Args:
            images (Tensor | Image): Image tensor or PIL Image.
            prompt (str): Prompt string.
            layer_wise (bool, optional): Whether to return layer-wise logits. Defaults to False.
        
        Returns:
            Tensor: Logits from the model, shape (vocab_size, batch, Optional[num_layers]).
        """
        pass
