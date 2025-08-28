# app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

# ----------------------
# CONFIG: Image & Models
# ----------------------
IMG_SIZE = 128

MODELS = {
    "Brain Tumor": {
        "path": "brain_tumor_model_weights.pth",  # weights only
        "categories": ["No Tumor", "Glioma", "Meningioma", "Pituitary"],
        "type": "brain"
    },
    "Lung Cancer": {
        "path": "lung_cnn_model_weights.pth",  # weights only
        "categories": ["Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Normal"],
        "type": "lung"
    }
}

# ----------------------
# NAVBAR: Select Application
# ----------------------
app_choice = st.selectbox("Choose Application", list(MODELS.keys()))
MODEL_PATH = MODELS[app_choice]["path"]
CATEGORIES = MODELS[app_choice]["categories"]
MODEL_TYPE = MODELS[app_choice]["type"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# DEFINE MODEL CLASSES
# ----------------------
class BrainCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128*14*14, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LungCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128*14*14, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ----------------------
# LOAD MODEL (weights only)
# ----------------------
@st.cache_resource
def load_trained_model(model_type, path, num_classes):
    if model_type == "brain":
        model = BrainCNN(num_classes=num_classes)
    else:
        model = LungCNN(num_classes=num_classes)

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_trained_model(MODEL_TYPE, MODEL_PATH, len(CATEGORIES))

# ----------------------
# IMAGE PREPROCESS
# ----------------------
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def predict_image(image: Image.Image):
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
    class_idx = int(np.argmax(probs))
    class_name = CATEGORIES[class_idx]
    confidence = float(np.max(probs)) * 100
    return class_name, confidence, probs

# ----------------------
# CUSTOM STYLING
# ----------------------
st.set_page_config(page_title="MediScan AI", page_icon="🩺", layout="wide")
st.markdown("""
    <style>
        .main { background: #f9fafb; }
        h1 { text-align: center; color: #4B0082; }
        .stRadio label { font-size: 18px !important; }
    </style>
""", unsafe_allow_html=True)

# ----------------------
# SESSION STATE
# ----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------
# SIDEBAR: Scan History
# ----------------------
st.sidebar.title("Scan History")
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []

if st.session_state.history:
    for entry in reversed(st.session_state.history[-10:]):
        st.sidebar.markdown(f"**🖼 {entry['File']}** → {entry['Prediction']} ({entry['Confidence']:.1f}%)")
else:
    st.sidebar.info("No scans yet.")

# ----------------------
# UI HEADER
# ----------------------
st.title(f" {app_choice} Classifier")
st.markdown(f"<p style='text-align:center;'>Upload medical scans to classify {app_choice.lower()} using Deep Learning.</p>", unsafe_allow_html=True)
st.markdown("---")

upload_type = st.radio("Choose input type:", ["📷 Single Image", "📂 Multiple Images"], horizontal=True)
results_data = []

# ----------------------
# SINGLE IMAGE MODE
# ----------------------
if upload_type == "📷 Single Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        class_name, confidence, prediction = predict_image(image)
        results_data.append({"File": uploaded_file.name, "Prediction": class_name, "Confidence (%)": round(confidence,2)})
        st.session_state.history.append({"File": uploaded_file.name, "Prediction": class_name, "Confidence": confidence})

        col1, col2 = st.columns([1,2])
        with col1:
            st.image(image, caption="🖼 Uploaded Image", use_container_width=True)
            st.markdown("### 🔎 Prediction Result")
            st.success(f"**{class_name}** ({confidence:.2f}%)")
        with col2:
            prob_df = pd.DataFrame({"Class": CATEGORIES, "Probability (%)": (prediction*100).round(2)})
            fig = px.bar(prob_df, x="Class", y="Probability (%)", color="Class", text="Probability (%)",
                         title="Prediction Confidence per Class", height=500)
            st.plotly_chart(fig, use_container_width=True)

# ----------------------
# MULTIPLE IMAGES MODE
# ----------------------
elif upload_type == "📂 Multiple Images":
    uploaded_files = st.file_uploader("Upload multiple images...", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        progress = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                image = Image.open(uploaded_file)
                class_name, confidence, _ = predict_image(image)
                results_data.append({"File": uploaded_file.name, "Prediction": class_name, "Confidence (%)": round(confidence,2)})
                st.session_state.history.append({"File": uploaded_file.name, "Prediction": class_name, "Confidence": confidence})
            except Exception as e:
                st.warning(f"Skipping {uploaded_file.name}: {e}")
            progress.progress((i+1)/len(uploaded_files))

        if results_data:
            df = pd.DataFrame(results_data)
            st.subheader("📊 Batch Prediction Results")
            st.dataframe(df, use_container_width=True, height=400)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results as CSV", csv, "batch_results.csv", "text/csv")

# ----------------------
# MAIN DASHBOARD ANALYTICS
# ----------------------
if results_data:
    st.markdown("---")
    st.markdown("### 📊 Data Analysis & Visualization")
    df = pd.DataFrame(results_data)

    # 1️⃣ Distribution Pie
    dist_df = df["Prediction"].value_counts().reset_index()
    dist_df.columns = ["Class", "Count"]
    st.plotly_chart(px.pie(dist_df, names="Class", values="Count", title=f"{app_choice} Distribution"), width=True)

    # 2️⃣ Heat Map
    heat_df = df.pivot(index="File", columns="Prediction", values="Confidence (%)")

    # 3️⃣ Top Confident Predictions
    top_df = df.sort_values(by="Confidence (%)", ascending=False).head(10)
    col1, col2 = st.columns([1,2])
    with col1:
        st.plotly_chart(px.imshow(heat_df, text_auto=True, title="Heat Map"), width=True)
    with col2:
        st.plotly_chart(px.bar(top_df, x="File", y="Confidence (%)", color="Prediction", title="Top Confident Predictions", text_auto=True), width=True)
