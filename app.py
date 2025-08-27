# app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

# ----------------------
# CONFIG
# ----------------------
IMG_SIZE = 128
CATEGORIES = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
MODEL_PATH = "brain_tumor_multiclass_model.h5"

# ----------------------
# LOAD MODEL
# ----------------------
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# ----------------------
# HELPERS
# ----------------------
def preprocess_image(image: Image.Image):
    img = image.convert("L").resize((IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

def predict_image(image: Image.Image):
    img = preprocess_image(image)
    prediction = model.predict(img, verbose=0)
    class_idx = np.argmax(prediction)
    class_name = CATEGORIES[class_idx]
    confidence = float(np.max(prediction)) * 100
    return class_name, confidence, prediction[0]


# ----------------------
# CUSTOM STYLING
# ----------------------
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="wide")
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
st.sidebar.title("📜 Scan History")
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
st.title("🧠 Brain Tumor Classifier")
st.markdown("<p style='text-align:center;'>Upload MRI scans to classify tumors using Deep Learning.</p>", unsafe_allow_html=True)
st.markdown("---")

upload_type = st.radio("Choose input type:", ["📷 Single Image", "📂 Multiple Images"], horizontal=True)

# ----------------------
# STORE RESULTS
# ----------------------
results_data = []

# ----------------------
# SINGLE IMAGE MODE
# ----------------------
if upload_type == "📷 Single Image":
    uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        class_name, confidence, prediction = predict_image(image)
        results_data.append({"File": uploaded_file.name, "Prediction": class_name, "Confidence (%)": round(confidence,2)})
        st.session_state.history.append({"File": uploaded_file.name, "Prediction": class_name, "Confidence": confidence})

        col1, col2 = st.columns([1,2])
        with col1:
            st.image(image, caption="🖼 Uploaded MRI", use_container_width=True)
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
    uploaded_files = st.file_uploader("Upload multiple MRI images...", type=["jpg","jpeg","png"], accept_multiple_files=True)
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

    # 1️⃣ Tumor Distribution Pie
    dist_df = df["Prediction"].value_counts().reset_index()
    dist_df.columns = ["Class", "Count"]
    st.plotly_chart(px.pie(dist_df, names="Class", values="Count", title="Tumor Distribution"), width=True)

    # 2️⃣ Heat Map
    heat_df = df.pivot(index="File", columns="Prediction", values="Confidence (%)")

    # 3️⃣ Top Confident Predictions
    top_df = df.sort_values(by="Confidence (%)", ascending=False).head(10)

    col1, col2 = st.columns([1,2])
    with col1:
        st.plotly_chart(px.imshow(heat_df, text_auto=True, title="Heat Map"), width=True)
    with col2:
        st.plotly_chart(px.bar(top_df, x="File", y="Confidence (%)", color="Prediction", title="Top Confident Predictions", text_auto=True), width=True)