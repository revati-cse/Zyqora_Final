import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetEngine:
    """Mac/development FP32 baseline; Pi runtime must be validated separately."""
    def __init__(self, use_pretrained=True):
        weights = MobileNet_V2_Weights.DEFAULT if use_pretrained else None
        self.model = mobilenet_v2(weights=weights).eval()
        self.categories = weights.meta["categories"] if weights else None
        self.model.to(torch.device("cpu"))

    @torch.inference_mode()
    def infer(self, tensor):
        return self.model(tensor.to("cpu"))

    def class_name(self, index):
        return self.categories[index] if self.categories else str(index)

_ENGINE = None
def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = MobileNetEngine()
    return _ENGINE
