import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# ================= CONFIG =================
DATASET_DIR = r"D:\Lauren_Projects\Ai-assisted radiology report generation\dataset1\data"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
MODEL_PATH = "densenet_normal_vs_pneumonia.pth"
TEST_SIZE = 0.2
# =========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ================= LOAD IMAGE PATHS =================
samples = []

for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
    cls_dir = os.path.join(DATASET_DIR, cls)
    for img in os.listdir(cls_dir):
        samples.append((os.path.join(cls_dir, img), label))

labels = [s[1] for s in samples]

# Train/Val split
train_samples, val_samples = train_test_split(
    samples,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=labels
)

# ================= DATASET =================
class XrayDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
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

train_ds = XrayDataset(train_samples)
val_ds   = XrayDataset(val_samples)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ================= MODEL =================
model = models.densenet121(weights="IMAGENET1K_V1")
model.classifier = nn.Linear(model.classifier.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ================= TRAIN =================
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

    # Validation
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
print("✅ Model saved:", MODEL_PATH)
