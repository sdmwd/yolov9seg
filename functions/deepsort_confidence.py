import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the YOLOv9 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='path/to/your/weights.pt', force_reload=True)

# Define the classes of objects to detect (damages on vehicles)
# Update this list with actual class names
classes = ['a', 'b', 'c']

# Open the video
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize DeepSORT object tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Dictionary to store best frames for each damage
extracted_damages = {}

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
        if conf.item() > 0.5 and cls in classes:  # Confidence threshold and class check
            bbox = [x1.item(), y1.item(), x2.item() -
                    x1.item(), y2.item() - y1.item()]
            detections.append((bbox, conf.item(), int(cls)))

    # Update tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    # Process tracked objects
    for track in tracked_objects:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        bbox = track.to_tlwh()
        x, y, w, h = map(int, bbox)

        # Save the best frame for each damage
        if track_id not in extracted_damages or extracted_damages[track_id]['score'] < track.score:
            extracted_damages[track_id] = {
                'frame_index': frame_index,
                'score': track.score,
                'bbox': bbox,
                'image': frame.copy()
            }

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# Save the extracted frames
output_dir = 'output'
for i, (obj_id, damage) in enumerate(extracted_damages.items()):
    if damage["image"] is not None:
        x, y, w, h = map(int, damage["bbox"])
        cv2.rectangle(damage["image"], (x, y), (x + w, y + h), (0, 255, 0), 2)
        output_image_path = f'{output_dir}/damage_{obj_id}.jpg'
        cv2.imwrite(output_image_path, damage["image"])
        print(f"Frame {damage["frame_index"]} saved as {output_image_path}")

print("Extraction of frames completed.")
