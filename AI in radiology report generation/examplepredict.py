import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============ CONFIG ============
MODEL_PATH = "densenet_normal_vs_pneumonia.pth"
IMAGE_PATH = "C:/Users/Nandhini.M/Downloads/pneumonia.png"
# ================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform (MUST MATCH TRAINING)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Load image
img = Image.open(IMAGE_PATH).convert("RGB")
img = transform(img).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    prob = torch.sigmoid(model(img)).item()

print("Pneumonia probability:", round(prob, 3))
print("Prediction:", "PNEUMONIA" if prob > 0.5 else "NORMAL")
