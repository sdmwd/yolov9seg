import os
import cv2
import json

main_dir = "chemin/vers/dossier_principal"
source_dir = os.path.join(main_dir, "sources")
boxes_dir = os.path.join(main_dir, "boxes")

os.makedirs(boxes_dir, exist_ok=True)

for image_file in os.listdir(source_dir):
    if image_file.endswith(".png") or image_file.endswith(".jpg"):
        image_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(source_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            continue

        # Création du dossier pour l'image dans le dossier des boîtes
        image_boxes_dir = os.path.join(boxes_dir, image_name)
        os.makedirs(image_boxes_dir, exist_ok=True)

        # Parcourir les classes (dossiers)
        for class_name in os.listdir(main_dir):
            class_dir = os.path.join(main_dir, class_name)
            if os.path.isdir(class_dir) and class_name not in ["sources", "boxes"]:
                mask_file = os.path.join(class_dir, image_file)

                if os.path.exists(mask_file):
                    mask = cv2.imread(mask_file, 0)

                    if mask is not None:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Créer un dossier pour la classe spécifique
                        class_boxes_dir = os.path.join(image_boxes_dir, class_name)
                        os.makedirs(class_boxes_dir, exist_ok=True)

                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            cropped_image = image[y:y+h, x:x+w]
                            cropped_mask = mask[y:y+h, x:x+w]

                            box_img_path = os.path.join(class_boxes_dir, f"{class_name}_{image_name}_box.jpg")
                            mask_img_path = os.path.join(class_boxes_dir, f"{class_name}_{image_name}_mask.png")

                            cv2.imwrite(box_img_path, cropped_image)
                            cv2.imwrite(mask_img_path, cropped_mask)

print("Préparation des images et masques recadrés terminée.")