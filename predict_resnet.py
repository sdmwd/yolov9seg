import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Step 1: Load the Model from the .pt File
best_model_path = 'best_model.pt'  # Path to the saved model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the entire model
model = torch.load(best_model_path, map_location=device)
model.eval()  # Set the model to evaluation mode

# Define the class names
class_names = ['class1', 'class2', 'class3', 'class4', 'class5']

# Step 2: Define the Image Transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Move the image to the device
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds[0].item()]
    
    return predicted_class
