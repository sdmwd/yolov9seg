import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Charger le modèle YOLOv9 pré-entraîné
net = cv2.dnn.readNet('weights/gelan-c.weights', 'cfg/gelan-c.cfg')

# Définir les classes d'objets à détecter (dommages sur les véhicules)
classes = ['scratch', 'dent', 'rust', ...]

# Initialiser le suivi d'objets avec DeepSort
deepsort = DeepSort("path/to/deepsort/model/weights")

# Ouvrir la vidéo
cap = cv2.VideoCapture('video.mp4')

# Initialiser un dictionnaire pour stocker les dommages extraits
extracted_damages = {}

while True:
    # Lire une frame de la vidéo
    ret, frame = cap.read()
    if not ret:
        break

    # Préparer le blob à partir de la frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Définir les sorties du réseau
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialiser les boîtes englobantes pour le suivi d'objets
    boxes = []
    scores = []
    class_ids = []

    # Parcourir les détections
    for output in outputs:
        for detection in output:
            detection_scores = detection[5:]
            class_id = np.argmax(detection_scores)
            confidence = detection_scores[class_id]

            # Vérifier si la détection est un dommage et si sa confiance est suffisamment élevée
            if classes[class_id] in ['scratch', 'dent', 'rust', ...] and confidence > 0.5:
                # Obtenir les coordonnées de la boîte englobante
                x, y, w, h = detection[0], detection[1], detection[2], detection[3]
                boxes.append([x, y, w, h])
                scores.append(confidence)
                class_ids.append(class_id)

    # Mettre à jour le suivi d'objets avec DeepSort
    tracked_objects = deepsort.update(boxes, scores, class_ids, frame)

    # Parcourir les objets suivis
    for x, y, w, h, obj_id, cls_id in tracked_objects:
        # Vérifier si ce dommage n'a pas déjà été extrait
        if obj_id not in extracted_damages:
            # Extraire la meilleure frame pour ce dommage
            roi = frame[y:y+h, x:x+w]
            extracted_damages[obj_id] = roi.copy()

            # Dessiner un rectangle autour du dommage
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Afficher la frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer la capture vidéo et les fenêtres
cap.release()
cv2.destroyAllWindows()

# Sauvegarder les frames extraites
for i, (obj_id, roi) in enumerate(extracted_damages.items()):
    cv2.imwrite(f'damages/{obj_id}_{i}.jpg', roi)