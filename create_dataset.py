import os
import shutil
import cv2
import random
from sklearn.model_selection import train_test_split

# Chemin du dossier principal
main_dir = "chemin/vers/dossier_principal"

# Dossiers cibles pour les images et les annotations
images_dir = os.path.join(main_dir, "images")
labels_dir = os.path.join(main_dir, "labels")
train_dir = os.path.join(main_dir, "train")
val_dir = os.path.join(main_dir, "val")
test_dir = os.path.join(main_dir, "test")

# Créer les dossiers si nécessaire
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)

# Fonction pour convertir les coordonnées de la bounding box au format YOLO
def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# Parcourir les fichiers dans le dossier source
source_dir = os.path.join(main_dir, "sources")
image_files = [f for f in os.listdir(source_dir) if f.endswith(".png") or f.endswith(".jpg")]

# Split des données
train_files, val_test_files = train_test_split(image_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(val_test_files, test_size=0.1, random_state=42)

# Fonction pour traiter et copier les fichiers
def process_files(files, split_dir):
    for image_file in files:
        image_path = os.path.join(source_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
        
        # Copier l'image dans le dossier approprié
        shutil.copy(image_path, os.path.join(split_dir, "images", image_file))
        
        # Fichier d'annotation YOLO
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(split_dir, "labels", label_file)
        
        with open(label_path, "w") as f:
            # Parcourir tous les dossiers (excepté 'sources') pour trouver les masques
            for folder_name in os.listdir(main_dir):
                folder_path = os.path.join(main_dir, folder_name)
                if os.path.isdir(folder_path) and folder_name != "sources":
                    mask_file = os.path.join(folder_path, image_file)
                    
                    if os.path.exists(mask_file):
                        mask = cv2.imread(mask_file, 0)  # Lire en tant qu'image en niveaux de gris
                        
                        # Identifier les contours du masque
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            bbox = convert_to_yolo(image.shape[1::-1], (x, y, x+w, y+h))
                            class_id = folder_name  # Utiliser le nom du dossier comme ID de classe
                            f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

# Traiter les différents splits
process_files(train_files, train_dir)
process_files(val_files, val_dir)
process_files(test_files, test_dir)

# Création du fichier YAML pour YOLOv9
yaml_content = f"""
train: {os.path.join(train_dir, 'images')}  # Chemin vers les images d'entraînement
val: {os.path.join(val_dir, 'images')}  # Chemin vers les images de validation
test: {os.path.join(test_dir, 'images')}  # Chemin vers les images de test

# Liste des noms de classes
names:
"""

# Récupérer toutes les classes (noms de dossiers sauf 'sources')
classes = [folder_name for folder_name in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, folder_name)) and folder_name != "sources"]

# Ajouter les classes au fichier YAML
for class_name in classes:
    yaml_content += f"  - {class_name}\n"

# Écrire le contenu dans un fichier dataset.yaml
yaml_path = os.path.join(main_dir, "dataset.yaml")
with open(yaml_path, "w") as yaml_file:
    yaml_file.write(yaml_content)

print(f"Fichier {yaml_path} créé avec succès.")