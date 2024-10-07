import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json


class CLEVR_generic(Dataset):
    def __init__(self, img_dir, transform=None) -> None:
        super().__init__()
        self.image_dir = img_dir
        if transform == None:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )
        self.transform = transform

        with open(img_dir + "/config.json", "r") as f:
            self.config = json.load(f)

    def __len__(self):
        return len(self.config["mapping"])

    def __getitem__(self, idx):
        filename = f"{idx}.png"
        img_path = os.path.join(self.image_dir, filename)
        position = self.config["mapping"][filename]
        if self.config["path_type"] == "translate":
            position = position[0]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, position
    
class CLEVR_generic_path(Dataset):
    def __init__(self, img_dir, transform=None) -> None:
        super().__init__()
        self.image_dir = img_dir
        with open(img_dir + "/config.json", "r") as f:
            self.config = json.load(f)

    def __len__(self):
        return len(self.config["mapping"])

    def __getitem__(self, idx):
        filename = f"{idx}.png"
        img_path = os.path.join(self.image_dir, filename)
        position = self.config["mapping"][filename]
        if self.config["path_type"] == "translate":
            position = position[0]
        return img_path, position


class CLEVR_Rotation(Dataset):
    """Deprecated. Use CLEVR_generic instead."""

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        if transform == None:
            transform = transforms.Compose([transforms.Resize((224, 224))])
        self.transform = transform
        self.img_annotations = []

        for filename in os.listdir(img_dir):
            if filename.endswith(".png"):
                angle = int(filename.split("_")[1].split(".")[0])
                self.img_annotations.append(
                    {"path": os.path.join(img_dir, filename), "angle": angle}
                )

        self.img_annotations.sort(key=lambda x: x["angle"])

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        img_path = self.img_annotations[idx]["path"]
        angle = self.img_annotations[idx]["angle"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, angle


class CLEVR_Translation(Dataset):
    """Deprecated. Use CLEVR_generic instead."""

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        if transform == None:
            transform = transforms.Compose([transforms.Resize((224, 224))])
        self.transform = transform
        self.img_annotations = []

        for filename in os.listdir(img_dir):
            if filename.endswith(".png"):
                translation = filename.split("translation_above_")[1].split(".png")[0]
                self.img_annotations.append(
                    {
                        "path": os.path.join(img_dir, filename),
                        "translation": translation,
                    }
                )

        self.img_annotations.sort(key=lambda x: x["translation"])

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        img_path = self.img_annotations[idx]["path"]
        translation = self.img_annotations[idx]["translation"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, translation
