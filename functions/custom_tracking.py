import cv2
import numpy as np
import torch
from supervision import Detections as BaseDetections, VideoInfo, process_video, BoundingBoxAnnotator
from supervision.config import CLASS_NAME_DATA_FIELD

# https://github.com/deepinvalue/yolov9-supervision-tracking-counting/blob/main/YOLOv9_Object_Detection_Tracking_and_Counting.ipynb

# Load the YOLOv9 model
model_path = 'path/to/your/weights.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=model_path, force_reload=True)

# Define the classes of objects to detect (damages on vehicles)
# Update this list with actual class names
classes = ['scratch', 'dent', 'rust']

# Open the video
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize a dictionary to store the best frames for each damage
extracted_damages = {}


class ExtendedDetections(BaseDetections):
    @classmethod
    def from_yolov9(cls, yolov9_results) -> 'ExtendedDetections':
        xyxy, confidences, class_ids = [], [], []

        for det in yolov9_results.pred:
            for *xyxy_coords, conf, cls_id in reversed(det):
                xyxy.append(torch.stack(xyxy_coords).cpu().numpy())
                confidences.append(float(conf))
                class_ids.append(int(cls_id))

        class_names = np.array([yolov9_results.names[i] for i in class_ids])

        if not xyxy:
            return cls.empty()

        return cls(
            xyxy=np.vstack(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
            data={CLASS_NAME_DATA_FIELD: class_names},
        )


def prepare_yolov9(model, conf=0.2, iou=0.7, classes=None, agnostic_nms=False, max_det=1000):
    model.conf = conf
    model.iou = iou
    model.classes = classes
    model.agnostic = agnostic_nms
    model.max_det = max_det
    return model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = prepare_yolov9(model, conf=0.2, iou=0.6, classes=[
                       classes.index(cls) for cls in classes])

video_info = VideoInfo.from_video_path(video_path)
annotator = BoundingBoxAnnotator(thickness=2)

# Implement a custom tracking class


class CustomTracker:
    def __init__(self):
        self.trackers = []

    def update(self, detections):
        # Example update function, replace with actual tracking logic
        new_trackers = []
        for det in detections:
            new_trackers.append(det)
        self.trackers = new_trackers
        return self.trackers


tracker = CustomTracker()

frame_index = 0


def annotate_frame(frame, index, video_info, detections, tracker, show_labels):
    tracked_objects = tracker.update(detections)
    for obj in tracked_objects:
        x1, y1, x2, y2, conf, cls = obj[:6]
        bbox = [x1, y1, x2 - x1, y2 - y1]
        if cls in classes:
            # Save the best frame for each damage
            if obj not in extracted_damages or extracted_damages[obj]['score'] < conf:
                extracted_damages[obj] = {
                    'frame_index': frame_index,
                    'score': conf,
                    'bbox': bbox,
                    'image': frame.copy()
                }
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)
    return frame


def process_video_with_tracker(model, tracker, video_path, output_path, classes):
    video_info = VideoInfo.from_video_path(video_path)
    model = prepare_yolov9(model, conf=0.2, iou=0.6, classes=[
                           classes.index(cls) for cls in classes])

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
        results = model(frame_rgb, size=640, augment=False)
        detections = ExtendedDetections.from_yolov9(results)
        return annotate_frame(frame, index, video_info, detections, tracker, show_labels=False)

    process_video(source_path=video_path,
                  target_path=output_path, callback=callback)


# Run the video processing
output_video_path = 'output_with_tracking.mp4'
process_video_with_tracker(model, tracker, video_path,
                           output_video_path, classes)

# Save the extracted frames
output_dir = 'output'
for i, (obj_id, damage) in enumerate(extracted_damages.items()):
    if damage['image'] is not None:
        x, y, w, h = map(int, damage['bbox'])
        cv2.rectangle(damage['image'], (x, y), (x + w, y + h), (0, 255, 0), 2)
        output_image_path = f'{output_dir}/damage_{obj_id}.jpg'
        cv2.imwrite(output_image_path, damage['image'])
        print(f"Frame {damage['frame_index']} saved as {output_image_path}")

print("Extraction of frames completed.")
