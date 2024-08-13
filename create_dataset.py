import os
import cv2
import json

# Chemin du dossier principal où se trouvent les images et les masques
main_dir = "chemin/vers/dossier_principal"
source_dir = os.path.join(main_dir, "sources")
labels_dir = os.path.join(main_dir, "labels")

# Création du dossier des labels s'il n'existe pas
os.makedirs(labels_dir, exist_ok=True)

for image_file in os.listdir(source_dir):
    if image_file.endswith(".png") or image_file.endswith(".jpg"):
        image_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(source_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        yolo_annotations = []

        # Parcourir les classes (dossiers)
        for class_name in os.listdir(main_dir):
            class_dir = os.path.join(main_dir, class_name)
            if os.path.isdir(class_dir) and class_name not in ["sources", "labels"]:
                mask_file = os.path.join(class_dir, image_file)
                
                if os.path.exists(mask_file):
                    mask = cv2.imread(mask_file, 0)  # Lire en tant qu'image en niveaux de gris
                    
                    if mask is not None:
                        # Trouver tous les contours, mais n'utiliser que le contour extérieur
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if contours:
                            # Combiner tous les contours en une seule grande forme et obtenir le bounding box global
                            all_contours = cv2.vstack(contours)
                            x, y, w, h = cv2.boundingRect(all_contours)

                            # Convertir les coordonnées du bounding box au format YOLO
                            x_center = (x + w / 2) / img_width
                            y_center = (y + h / 2) / img_height
                            width = w / img_width
                            height = h / img_height

                            # Assigner un ID de classe basé sur le nom du dossier (vous pouvez ajuster cela)
                            class_id = 0  # Remplacez cela si vous avez plusieurs classes ou un mapping spécifique

                            # Ajouter l'annotation dans le format YOLO
                            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

        # Écrire les annotations YOLO dans un fichier texte
        if yolo_annotations:
            label_file_path = os.path.join(labels_dir, f"{image_name}.txt")
            with open(label_file_path, "w") as f:
                f.write("\n".join(yolo_annotations))

print("Préparation des annotations YOLOv9 terminée.")