# https://github.com/ultralytics/ultralytics/issues/1310


import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import cv2
import numpy as np

def process_image_in_batch(image_str, device, fp16):
    # Decode base64 image
    b_img = np.frombuffer(base64.b64decode(image_str.encode('ascii')), dtype=np.uint8)
    im = cv2.imdecode(b_img, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Normalize and convert to tensor
    im = torch.from_numpy(im).to(device)
    im = im.permute(2, 0, 1)  # Change to (C, H, W) format
    im = im.half() if fp16 else im.float()
    im /= 255.0
    if len(im.shape) == 3:
        im = im.unsqueeze(0)  # Add batch dimension if needed
    return im

def load_images_in_batches(data, batch_size, device, fp16):
    images = data['base64']
    batch = []
    futures = []
    with ThreadPoolExecutor() as executor:
        for img_str in images:
            future = executor.submit(process_image_in_batch, img_str, device, fp16)
            futures.append(future)
            if len(futures) == batch_size:
                for future in as_completed(futures):
                    batch.append(future.result())
                yield torch.cat(batch, 0)  # Combine images into a single batch tensor
                batch = []
                futures = []
        if futures:
            for future in as_completed(futures):
                batch.append(future.result())
            yield torch.cat(batch, 0)

# Example usage in the run function
def run(
    weights,
    data,
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device='',
    save_txt=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project='runs/predict-seg',
    name='exp',
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
    retina_masks=False,
    batch_size=16
):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data='data/coco.yaml', fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    model.warmup(imgsz=(batch_size, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for batch in load_images_in_batches(data, batch_size, device, half):
        with dt[0]:
            batch = batch.to(device)
            batch = batch.half() if model.fp16 else batch.float()
            batch /= 255.0

        with dt[1]:
            pred, proto = model(batch, augment=augment, visualize=visualize)[:2]

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)


    
def predict(self, data: dict) -> dict:


    batch_imgs = load_images_in_batches(data, batch_size=16, device=device, fp16=False)
    results = []

    for batch in batch_imgs:
        with torch.no_grad():
            output = model(batch)
        results.append(output.cpu().numpy())

    return {
        'scores': [item.tolist() for sublist in results for item in sublist],
        'classes': list(range(len(results)))
    }





import torch
import cv2
import base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Function to process a single image
def process_image(image_str: str):
    b_img = np.frombuffer(base64.b64decode(image_str.encode('ascii')), dtype=np.uint8)
    im = cv2.imdecode(b_img, cv2.IMREAD_COLOR)
    im = vges_transform(im)  # Apply your transform which includes T.Compose([Resize, ToTensor])
    return im

# Function to process a batch of images in parallel
def process_batch(images: list):
    with ThreadPoolExecutor() as executor:
        batch_imgs = list(executor.map(process_image, images))
    batch_imgs = torch.stack(batch_imgs)  # Stack images into a single tensor
    return batch_imgs

class vges_RequestHandler:
    def __init__(self):
        pass

    def predict(self, data: dict) -> dict:
        assert 'base64' in data, 'Invalid arguments: "base64" needed'
        global vges_model
        global vges_transform
        global vges_device

        # Reading and processing images received in base64
        images = data['base64']
        batch_imgs = process_batch(images)

        print(f'Batch images shape: {batch_imgs.shape}')  # Debug: Check the shape

        # Convert to tensor and move to device
        x = batch_imgs.to(vges_device)
        x = x.float() / 255.0  # Normalize
        
        print(f'Tensor shape before model: {x.shape}')  # Debug: Check the shape

        # Run inference on the entire batch
        with torch.no_grad():
            output = vges_model(x)

        print(f'Output shape: {output.shape}')  # Debug: Check the shape

        # Construct the return object
        result = {
            'scores': output.cpu().numpy().tolist(),
            'classes': list(range(output.shape[0]))
        }

        return result

# Initialize and load model
vges_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vges_model = torch.nn.Sequential(
    # Example model architecture
    torch.nn.Linear(512, 10),  # Modify as needed
    torch.nn.Sigmoid()
)
vges_model = vges_model.to(vges_device)
vges_model.eval()

# Example usage
if __name__ == "__main__":
    handler = vges_RequestHandler()
    data = {
        'base64': [
            # list of base64 encoded image strings
        ]
    }
    results = handler.predict(data)
    print(results)