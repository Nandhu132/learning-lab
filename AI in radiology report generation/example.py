import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

TRAIN_DIR = "D:/Lauren_Projects/Ai-assisted radiology report generation/data/chest_xray/train"
VAL_DIR   = "D:/Lauren_Projects/Ai-assisted radiology report generation/data/chest_xray/val"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
MODEL_PATH = "normal_vs_pneumonia.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
            cls_dir = os.path.join(root_dir, cls)
            for img in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, img), label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.float32)
        return img, label

train_ds = PneumoniaDataset(TRAIN_DIR)
val_ds   = PneumoniaDataset(VAL_DIR)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = models.densenet121(weights="IMAGENET1K_V1")
model.classifier = nn.Linear(model.classifier.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )

torch.save(model.state_dict(), MODEL_PATH)
print(" Model saved:", MODEL_PATH)
