"""Multi-object detection (YOLO) for mixed-waste images.

The detector proposes object boxes; each box is cropped and classified by the
existing MobileNetV2 waste model (zyqora.mixed). The detector's own class label
(COCO-80 for the shipped pretrained weights) is used only as a *material hint*
through a validated, conservative mapping (config/detector_materials.yaml); it is
never presented as a waste-category prediction.
"""

from .yolo import (
    Detection,
    DetectorConfig,
    DetectorError,
    DetectorOutput,
    YoloDetector,
    letterbox,
    load_detector_config,
    nms,
)

__all__ = [
    "Detection", "DetectorConfig", "DetectorError", "DetectorOutput", "YoloDetector",
    "letterbox", "load_detector_config", "nms",
]
