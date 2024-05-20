import argparse
import os
import platform
import sys
from pathlib import Path
import base64
import torch
import numpy as np
import cv2
from typing import Union, Tuple

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (Profile, check_img_size, check_requirements, non_max_suppression, scale_boxes, letterbox)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Function to load the model
def load_model(
    weights: Union[str, Path],
    img_size: Tuple[int, int] = (640, 640),
    device: str = 'cpu'
):
    global imgsz
    imgsz = img_size

    global vehicle_device
    vehicle_device = select_device(device)
    print(f"Running on: {vehicle_device}")

    global model
    model = DetectMultiBackend(weights, device=vehicle_device, dnn=False, data='datasets.yaml', fp16=False)
    model.warmup(imgsz=(1, 3, *imgsz))  # Important for reducing first inference time

# Function to process a single image
def process_image(image_str: str, imgsz):
    b_img = np.frombuffer(base64.b64decode(image_str.encode('ascii')), dtype=np.uint8)
    im = cv2.imdecode(b_img, cv2.IMREAD_COLOR)
    im0 = im.copy()
    im = letterbox(im, new_shape=imgsz, auto=True)[0]
    im = np.moveaxis(im, -1, 0)  # Change to (C, H, W)
    im = np.ascontiguousarray(im)
    return im, im0

# Function to process a batch of images
def process_batch(images: list, imgsz):
    batch_imgs = []
    batch_imgs0 = []
    original_sizes = []
    for img_str in images:
        im, im0 = process_image(img_str, imgsz)
        batch_imgs.append(im)
        batch_imgs0.append(im0)
        original_sizes.append(im0.shape[:2])  # Store original sizes
    batch_imgs = np.stack(batch_imgs)
    return batch_imgs, batch_imgs0, original_sizes

# Function to perform batch inference
def batch_inference(batch_imgs, model, vehicle_device):
    im = torch.from_numpy(batch_imgs).to(vehicle_device)
    im = im.half() if model.fp16 else im.float()
    im /= 255.0  # Normalize
    with torch.no_grad():
        pred = model(im, augment=False, visualize=False)
    return pred

# Function to perform non-max suppression on batch
def batch_nms(pred):
    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 1000
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det, nm=32)
    return pred

# Main request handler class
class VehicleSegmentation_RequestHandler:
    def __init__(self):
        pass

    def predict(self, data: dict) -> dict:
        assert 'base64' in data, 'Invalid arguments: "base64" needed'
        global vehicle_device
        global imgsz
        global model
        stride = model.stride

        # Reading and processing images received in base64
        images = data['base64']
        batch_imgs, batch_imgs0, original_sizes = process_batch(images, imgsz)

        # Run inference
        pred = batch_inference(batch_imgs, model, vehicle_device)

        # Process NMS
        pred = batch_nms(pred)

        # Process predictions
        results = []
        for i, det in enumerate(pred):  # per image
            im0 = batch_imgs0[i]
            annotator = Annotator(im0, line_width=3, example=str(model.names))
            if len(det):
                det[:, :4] = scale_boxes(batch_imgs[i].shape[1:], det[:, :4], original_sizes[i]).round()
                
                # Collect results for each detection
                bboxes = det[:, :4].tolist()
                scores = det[:, 4].tolist()
                classes = det[:, 5].tolist()
                
                results.append({
                    'bboxes': bboxes,
                    'scores': scores,
                    'classes': classes
                })

                for *xyxy, conf, cls in reversed(det[:, :6]):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(cls, True))

            im0 = annotator.result()

        return results

# Initialize and load model
weights_path = "path/to/yolo-seg.pt"
load_model(weights=weights_path, img_size=(640, 640), device='cuda')

# Example usage
handler = VehicleSegmentation_RequestHandler()
data = {
    'base64': [
        # list of base64 encoded image strings
    ]
}
results = handler.predict(data)
print(results)