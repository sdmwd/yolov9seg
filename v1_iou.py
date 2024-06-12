import cv2
import torch
import numpy as np
from collections import defaultdict

# Charger le modèle YOLOv9 gelan-c
model_path = 'chemin/vers/tes/poids.pt'
model = torch.hub.load('gelan/yolov9', 'gelan-c', path=model_path, force_reload=True)

# Ouvrir la vidéo
video_path = 'chemin/vers/ta/vidéo.mp4'
cap = cv2.VideoCapture(video_path)

# Dictionnaire pour stocker les dommages détectés
damages = defaultdict(lambda: {"frame_index": None, "score": 0, "box": None, "image": None})

# Fonction pour calculer l'IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

frame_index = 0
iou_threshold = 0.5  # Seuil pour considérer deux boxes comme représentant le même dommage
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer le modèle YOLOv9
    results = model(frame)

    # Parcourir les prédictions
    for pred in results.pred[0]:
        x1, y1, x2, y2, conf, cls = pred[:6]
        box = (x1.item(), y1.item(), (x2 - x1).item(), (y2 - y1).item())

        # Vérifier si le dommage détecté correspond à un dommage existant
        match_found = False
        for damage_key, damage in damages.items():
            if calculate_iou(box, damage["box"]) > iou_threshold:
                match_found = True
                # Mettre à jour les informations de dommage si score de confiance plus élevé
                if conf.item() > damage["score"]:
                    damages[damage_key] = {
                        "frame_index": frame_index,
                        "score": conf.item(),
                        "box": box,
                        "image": frame.copy()
                    }
                break

        # Si aucun match trouvé, ajouter un nouveau dommage
        if not match_found:
            damage_key = len(damages)
            damages[damage_key] = {
                "frame_index": frame_index,
                "score": conf.item(),
                "box": box,
                "image": frame.copy()
            }

    frame_index += 1

# Libérer les ressources
cap.release()

# Sauvegarder les images avec les dommages les plus visibles
output_dir = 'output'
for i, damage in enumerate(damages.values()):
    if damage["image"] is not None:
        x, y, w, h = map(int, damage["box"])
        cv2.rectangle(damage["image"], (x, y), (x + w, y + h), (0, 255, 0), 2)
        output_image_path = f'{output_dir}/damage_{i}.jpg'
        cv2.imwrite(output_image_path, damage["image"])
        print(f"Frame {damage['frame_index']} saved as {output_image_path}")

print("Extraction des frames terminée.")
