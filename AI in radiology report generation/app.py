import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "densenet121_chestxray.pth"
THRESHOLD = 0.5

LABEL_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Chest X-ray Disease Detection",
    layout="wide"
)

st.title(" Chest X-ray Disease Prediction")
st.write("Upload a chest X-ray image to predict possible diseases.")

@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABEL_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")
    col1.image(image, caption="Uploaded X-ray", width="stretch")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    col2.subheader("Disease Probabilities")

    results = list(zip(LABEL_COLS, probs))
    results.sort(key=lambda x: x[1], reverse=True)

    for label, prob in results:
        col2.progress(float(prob), text=f"{label}: {prob:.2f}")

    st.subheader(" Predicted Diseases ")

    predicted = [label for label, prob in results if prob >= THRESHOLD]

    if predicted:
        for disease in predicted:
            st.success(disease)
    else:
        st.info("No disease detected.")
