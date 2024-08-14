import os
import cv2
import json

# Chemin du dossier principal
main_dir = "path/to/your/main_folder"
source_dir = os.path.join(main_dir, "photos", "sources")
output_dir = os.path.join(main_dir, "sam_finetune_dataset")

# Dossiers de sortie pour les images et masques recadrés
cropped_images_dir = os.path.join(output_dir, "cropped_images")
cropped_masks_dir = os.path.join(output_dir, "cropped_masks")
annotations_dir = os.path.join(output_dir, "annotations")

# Création des dossiers de sortie si nécessaire
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(cropped_images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(cropped_masks_dir, split), exist_ok=True)
    os.makedirs(os.path.join(annotations_dir, split), exist_ok=True)

# Parcours des images dans le répertoire source
for split in ['train', 'val', 'test']:
    image_split_dir = os.path.join(source_dir, split)
    
    for image_file in os.listdir(image_split_dir):
        if image_file.endswith(".png") or image_file.endswith(".jpg"):
            image_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(image_split_dir, image_file)
            image = cv2.imread(image_path)
            img_height, img_width = image.shape[:2]

            # Charger les annotations YOLOv9 pour obtenir les bounding boxes
            yolo_annotation_path = os.path.join(main_dir, "labels", split, f"{image_name}.txt")
            
            if not os.path.exists(yolo_annotation_path):
                print(f"Aucune annotation trouvée pour {image_file}.")
                continue

            with open(yolo_annotation_path, "r") as file:
                annotations = file.readlines()

            for idx, annotation in enumerate(annotations):
                class_id, x_center, y_center, width, height = map(float, annotation.split())
                x_center = int(x_center * img_width)
                y_center = int(y_center * img_height)
                box_width = int(width * img_width)
                box_height = int(height * img_height)
                
                x_min = max(x_center - box_width // 2, 0)
                y_min = max(y_center - box_width // 2, 0)
                x_max = min(x_center + box_width // 2, img_width)
                y_max = min(y_center + box_width // 2, img_height)
                
                # Recadrer l'image selon la bounding box
                cropped_image = image[y_min:y_max, x_min:x_max]
                cropped_image_filename = f"{image_name}_box_{idx}.jpg"
                cv2.imwrite(os.path.join(cropped_images_dir, split, cropped_image_filename), cropped_image)

                # Recherche du masque correspondant
                found_mask = False
                for class_name in os.listdir(os.path.join(main_dir, "photos")):
                    class_dir = os.path.join(main_dir, "photos", class_name)
                    if os.path.isdir(class_dir) and class_name != "sources":
                        mask_file = os.path.join(class_dir, split, image_file)
                        
                        if os.path.exists(mask_file):
                            mask = cv2.imread(mask_file, 0)
                            if mask is not None:
                                # Recadrer le masque selon la bounding box
                                cropped_mask = mask[y_min:y_max, x_min:x_max]
                                cropped_mask_filename = f"{image_name}_mask_{idx}.png"
                                cv2.imwrite(os.path.join(cropped_masks_dir, split, cropped_mask_filename), cropped_mask)
                                
                                # Créer le fichier d'annotation JSON pour SAM2
                                annotation_data = {
                                    "image": cropped_image_filename,
                                    "mask": cropped_mask_filename,
                                    "bbox": [x_min, y_min, x_max, y_max],
                                    "class_id": int(class_id),
                                    "class_name": class_name
                                }
                                annotation_file_path = os.path.join(annotations_dir, split, f"{image_name}_box_{idx}.json")
                                with open(annotation_file_path, "w") as ann_file:
                                    json.dump(annotation_data, ann_file)
                                found_mask = True
                                break

                if not found_mask:
                    print(f"Aucun masque trouvé pour {image_file} avec la bbox {idx}.")

print("Préparation du dataset pour le fine-tuning de SAM2 terminée.")