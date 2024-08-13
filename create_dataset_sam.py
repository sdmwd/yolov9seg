import os
import cv2
import json

# Directory paths
main_dir = "path/to/your/dataset"
images_dir = os.path.join(main_dir, "images")
labels_dir = os.path.join(main_dir, "labels")
cropped_images_dir = os.path.join(main_dir, "cropped_images")
cropped_masks_dir = os.path.join(main_dir, "cropped_masks")
annotations_dir = os.path.join(main_dir, "annotations")

# Create necessary directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(cropped_images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(cropped_masks_dir, split), exist_ok=True)
    os.makedirs(os.path.join(annotations_dir, split), exist_ok=True)

for split in ['train', 'val', 'test']:
    image_split_dir = os.path.join(images_dir, split)
    label_split_dir = os.path.join(labels_dir, split)
    
    for image_file in os.listdir(image_split_dir):
        image_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_split_dir, image_file)
        label_path = os.path.join(label_split_dir, f"{image_name}.txt")
        
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]

        with open(label_path, "r") as file:
            annotations = file.readlines()
        
        for idx, annotation in enumerate(annotations):
            class_id, x_center, y_center, width, height = map(float, annotation.split())
            x_center = int(x_center * img_width)
            y_center = int(y_center * img_height)
            box_width = int(width * img_width)
            box_height = int(height * img_height)
            
            x_min = max(x_center - box_width // 2, 0)
            y_min = max(y_center - box_height // 2, 0)
            x_max = min(x_center + box_width // 2, img_width)
            y_max = min(y_center + box_height // 2, img_height)
            
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_image_filename = f"{image_name}_box_{idx}.jpg"
            cv2.imwrite(os.path.join(cropped_images_dir, split, cropped_image_filename), cropped_image)
            
            # Assuming the mask follows the same naming convention
            mask_path = os.path.join(images_dir.replace('images', 'masks'), split, image_file)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0)
                cropped_mask = mask[y_min:y_max, x_min:x_max]
                cropped_mask_filename = f"{image_name}_mask_{idx}.png"
                cv2.imwrite(os.path.join(cropped_masks_dir, split, cropped_mask_filename), cropped_mask)

                # Create annotation JSON
                annotation_data = {
                    "image": cropped_image_filename,
                    "mask": cropped_mask_filename,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "class_id": int(class_id)
                }
                annotation_file_path = os.path.join(annotations_dir, split, f"{image_name}_box_{idx}.json")
                with open(annotation_file_path, "w") as ann_file:
                    json.dump(annotation_data, ann_file)
