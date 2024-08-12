import os
import cv2
import json

# Chemin du dossier principal où se trouvent les images et les masques
main_dir = "chemin/vers/dossier_principal"
source_dir = os.path.join(main_dir, "sources")
output_dir = os.path.join(main_dir, "sam_finetune_dataset")

# Dossiers de sortie
cropped_images_dir = os.path.join(output_dir, "images")
cropped_masks_dir = os.path.join(output_dir, "masks")
annotations_dir = os.path.join(output_dir, "annotations")

# Création des dossiers de sortie si nécessaire
os.makedirs(cropped_images_dir, exist_ok=True)
os.makedirs(cropped_masks_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

for image_file in os.listdir(source_dir):
    if image_file.endswith(".png") or image_file.endswith(".jpg"):
        image_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(source_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            continue

        # Chargement des annotations YOLOv9 pour obtenir les bounding boxes
        yolo_annotation_path = os.path.join(main_dir, "labels", f"{image_name}.txt")
        
        if not os.path.exists(yolo_annotation_path):
            print(f"Aucune annotation trouvée pour {image_file}.")
            continue
        
        with open(yolo_annotation_path, "r") as file:
            yolo_annotations = file.readlines()

        # Itération sur chaque bounding box dans les annotations YOLO
        for idx, annotation in enumerate(yolo_annotations):
            class_id, x_center, y_center, width, height = map(float, annotation.split())
            
            img_height, img_width = image.shape[:2]
            x_center = int(x_center * img_width)
            y_center = int(y_center * img_height)
            box_width = int(width * img_width)
            box_height = int(height * img_height)
            
            x_min = max(x_center - box_width // 2, 0)
            y_min = max(y_center - box_height // 2, 0)
            x_max = min(x_center + box_width // 2, img_width)
            y_max = min(y_center + box_height // 2, img_height)
            
            # Recadrer la région d'intérêt de l'image
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_image_filename = f"{image_name}_box_{idx}.jpg"
            cv2.imwrite(os.path.join(cropped_images_dir, cropped_image_filename), cropped_image)
            
            # Identifier et recadrer le masque correspondant
            for class_name in os.listdir(main_dir):
                class_dir = os.path.join(main_dir, class_name)
                if os.path.isdir(class_dir) and class_name not in ["sources", "labels", "sam_finetune_dataset"]:
                    mask_file = os.path.join(class_dir, image_file)
                    
                    if os.path.exists(mask_file):
                        mask = cv2.imread(mask_file, 0)  # Lire en niveaux de gris
                        
                        if mask is not None:
                            cropped_mask = mask[y_min:y_max, x_min:x_max]
                            cropped_mask_filename = f"{image_name}_mask_{idx}.png"
                            cv2.imwrite(os.path.join(cropped_masks_dir, cropped_mask_filename), cropped_mask)

                            # Créer une annotation pour SAM
                            annotation_data = {
                                "image": cropped_image_filename,
                                "mask": cropped_mask_filename,
                                "bounding_box": [x_min, y_min, x_max, y_max],
                                "class_id": int(class_id),
                                "class_name": class_name
                            }
                            annotation_file = os.path.join(annotations_dir, f"{image_name}_box_{idx}.json")
                            with open(annotation_file, "w") as ann_file:
                                json.dump(annotation_data, ann_file)
        
print("Préparation du dataset pour le fine-tuning de SAM terminée.")