# pip install torch opencv-python numpy
# pip install bytetrack



import cv2
import torch
import numpy as np
from bytetrack import BYTETracker

# Load YOLOv9 model
model_path = 'path/to/your/weights.pt'
model = torch.hub.load('gelan/yolov9', 'gelan-c', path=model_path, force_reload=True)

# Initialize video capture
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize ByteTrack tracker
tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8)

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

    # Convert detections to ByteTrack format
    dets = np.array([[*d[0], d[1], d[2]] for d in detections])

    # Update tracker
    tracks = tracker.update(dets, frame.shape[:2])

    for track in tracks:
        if not track.is_confirmed():
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
