git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2



pip install -e .
pip install -e ".[demo]"  # For additional demo dependencies


  pip install torch torchvision


  cd checkpoints
./download_ckpts.sh
cd ..


  Alternatively, you can download specific checkpoints:

SAM2 Tiny
SAM2 Small
SAM2 Base Plus
SAM2 Large

  dataset/
├── cropped_images/
│   ├── train/
│   ├── val/
│   └── test/
├── cropped_masks/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train/
    ├── val/
    └── test/



  import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load the model and checkpoint
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, checkpoint, device='cuda')

# Prepare the predictor
predictor = SAM2ImagePredictor(sam2_model)

# Load your dataset
train_data = load_your_dataset("dataset/cropped_images/train")
val_data = load_your_dataset("dataset/cropped_images/val")

# Example of setting the image and fine-tuning the model
for image, mask in train_data:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        predictions = predictor.predict(prompt=mask)
        # Train your model here using the predictions and ground truth
        # Backpropagation, optimizer step, etc.

# Save the fine-tuned model
torch.save(sam2_model.state_dict(), "fine_tuned_sam2_model.pth")



with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(new_image)
    masks, _, _ = predictor.predict(prompt=bbox)
