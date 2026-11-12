from PIL import Image, ImageOps
import torch
from torchvision import transforms

_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

def load_image(path: str):
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
        return _PREPROCESS(image).unsqueeze(0)

def horizontal_flip(tensor: torch.Tensor):
    return torch.flip(tensor, dims=[3]).contiguous()
