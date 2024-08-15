import torch
from torch.utils.data import DataLoader, Dataset
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch.nn.functional as F
import torchvision.transforms as transforms
import json
from PIL import Image

# Custom Dataset
class SAM2Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, annotations_dir, split):
        self.images_dir = os.path.join(images_dir, split)
        self.masks_dir = os.path.join(masks_dir, split)
        self.annotations_dir = os.path.join(annotations_dir, split)
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        annotation_name = image_name.replace(".jpg", ".json").replace(".png", ".json")
        annotation_path = os.path.join(self.annotations_dir, annotation_name)
        
        with open(annotation_path, "r") as f:
            annotation = json.load(f)

        image = Image.open(image_path).convert("RGB")
        image = transforms.ToTensor()(image)

        mask_path = os.path.join(self.masks_dir, annotation["mask"])
        mask = Image.open(mask_path).convert("L")
        mask = transforms.ToTensor()(mask)

        bbox = annotation["bbox"]
        class_id = annotation["class_id"]

        return image, mask, bbox, class_id

# Load the dataset
main_dir = "path/to/your/sam_finetune_dataset"
train_dataset = SAM2Dataset(os.path.join(main_dir, "cropped_images"), os.path.join(main_dir, "cropped_masks"), os.path.join(main_dir, "annotations"), "train")
val_dataset = SAM2Dataset(os.path.join(main_dir, "cropped_images"), os.path.join(main_dir, "cropped_masks"), os.path.join(main_dir, "annotations"), "val")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load the SAM2 model
checkpoint = "path/to/sam2_t.pt"
model_cfg = "path/to/sam2_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, checkpoint, device='cuda')

# Modify the model to output class predictions
sam2_model.classifier = torch.nn.Linear(sam2_model.config.hidden_size, num_classes)  # Add a classification head if needed

# Training loop
optimizer = torch.optim.Adam(sam2_model.parameters(), lr=1e-4)
segmentation_loss = torch.nn.BCELoss()  # For segmentation
classification_loss = torch.nn.CrossEntropyLoss()  # For classification

sam2_model.train()
for epoch in range(10):  # Adjust the number of epochs
    epoch_loss = 0
    for images, masks, bboxes, class_ids in train_loader:
        images, masks, class_ids = images.cuda(), masks.cuda(), class_ids.cuda()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Mixed precision training
            predictor.set_image(images)
            predictions = predictor.predict(prompts=bboxes)  # Generate predictions

            # Compute losses
            seg_loss = segmentation_loss(predictions, masks)
            cls_predictions = sam2_model.classifier(predictor.model_output)  # Assuming the model output has a classifier head
            cls_loss = classification_loss(cls_predictions, class_ids)

            loss = seg_loss + cls_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

    # Validation loop (optional)
    sam2_model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks, bboxes, class_ids in val_loader:
            images, masks, class_ids = images.cuda(), masks.cuda(), class_ids.cuda()

            with torch.cuda.amp.autocast():
                predictor.set_image(images)
                predictions = predictor.predict(prompts=bboxes)
                seg_loss = segmentation_loss(predictions, masks)
                cls_predictions = sam2_model.classifier(predictor.model_output)
                cls_loss = classification_loss(cls_predictions, class_ids)

                loss = seg_loss + cls_loss
                val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader)}")
    sam2_model.train()

# Save the fine-tuned model
torch.save(sam2_model.state_dict(), "fine_tuned_sam2_model.pth")
