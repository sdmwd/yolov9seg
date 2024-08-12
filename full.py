from yolov9 import YOLO  # Hypothetical import, adjust based on your actual YOLO implementation

yolo_model = YOLO("path/to/your/yolov9/weights.pt")
image = cv2.imread("path/to/new/image.jpg")

detections = yolo_model.detect(image)  # This will return bounding boxes


from sam import SAM  # Hypothetical import, adjust based on your actual SAM implementation

sam_model = SAM("path/to/sam/weights.pt")

for bbox in detections:
    x, y, w, h = bbox  # Assuming bbox is [x, y, w, h]
    cropped_region = image[y:y+h, x:x+w]

    # Use SAM to segment this region
    mask = sam_model.segment(cropped_region)

    # Combine the mask with the original image or save it
    # Example: Save the mask
    cv2.imwrite(f"output_masks/mask_{x}_{y}.png", mask)