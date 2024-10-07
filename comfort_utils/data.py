import os
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CocoSegDataset(Dataset):
    def __init__(self, seg_json_file, img_dir):
        with open(seg_json_file, "r") as f:
            self.img_annotations = [json.loads(line) for line in f]
        self.img_dir = img_dir
        self.transform = transforms.Compose([transforms.Resize((224, 224))])

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_annotations[idx]["image"])
        image = Image.open(img_path).convert("RGB")
        objects = self.img_annotations[idx]["objects"]

        if self.transform:
            image = self.transform(image)

        return image, objects
