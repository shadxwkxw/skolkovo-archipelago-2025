import numpy as np
from typing import List, Union
from ultralytics import YOLO
import torch

model = YOLO("best.pt")


def infer_image_bbox(image: np.ndarray) -> List[dict]:
    result_list = []

    results = model.predict(
        source=image,
        imgsz=1920,
        conf=0.25,
        iou=0.45,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    for r in results:
        boxes = r.boxes
        for box in boxes:
            xywhn = box.xywhn[0].cpu().numpy()
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            result_list.append({
                "xc": round(float(xywhn[0]), 6),
                "yc": round(float(xywhn[1]), 6),
                "w": round(float(xywhn[2]), 6),
                "h": round(float(xywhn[3]), 6),
                "label": cls,
                "score": round(conf, 6)
            })

    return result_list


def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    if isinstance(images, np.ndarray):
        images = [images]

    return [infer_image_bbox(image) for image in images]
