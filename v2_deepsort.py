# pip install torch opencv-python numpy scikit-image
# pip install deep_sort_realtime

# https://github.com/sujanshresstha/YOLOv9_DeepSORT


import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv9 model
model_path = 'path/to/your/weights.pt'
model = torch.hub.load('gelan/yolov9', 'gelan-c', path=model_path, force_reload=True)

# Initialize video capture
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Dictionary to store best frames for each damage
damages = {}

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply YOLOv9 model
    results = model(frame)
    detections = []

    # Parse detections
    for pred in results.pred[0]:
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf.item() > 0.5:  # Confidence threshold
            bbox = [x1.item(), y1.item(), x2.item() - x1.item(), y2.item() - y1.item()]
            detections.append((bbox, conf.item(), cls.item()))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        bbox = track.to_tlwh()
        x, y, w, h = map(int, bbox)

        # Save the best frame for each damage
        if track_id not in damages or damages[track_id]['score'] < track.score:
            damages[track_id] = {
                'frame_index': frame_index,
                'score': track.score,
                'bbox': bbox,
                'image': frame.copy()
            }

    frame_index += 1

cap.release()

# Save the best frames
output_dir = 'output'
for i, damage in enumerate(damages.values()):
    if damage["image"] is not None:
        x, y, w, h = map(int, damage["bbox"])
        cv2.rectangle(damage["image"], (x, y), (x + w, y + h), (0, 255, 0), 2)
        output_image_path = f'{output_dir}/damage_{i}.jpg'
        cv2.imwrite(output_image_path, damage["image"])
        print(f"Frame {damage['frame_index']} saved as {output_image_path}")

print("Extraction of frames completed.")
