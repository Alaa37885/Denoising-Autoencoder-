import streamlit as st
import base64
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="MNIST Autoencoder - Image Denoising",
    layout="wide"
)

# =========================
# BACKGROUND IMAGE
# =========================
img_path = r"D:\Material of level 3\Semi 2\Applied ML\Denoising_Autoencoder\photo.jpg"

def set_bg(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>

    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.55);
        z-index: 0;
    }}

    .main {{
        position: relative;
        z-index: 1;
    }}

    h1, h2, h3, p {{
        color: white !important;
    }}

    /* Upload Box */
    div[data-testid="stFileUploader"] section {{
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px;
        padding: 10px;
    }}

    div[data-testid="stFileUploader"] label {{
        color: #ddd !important;
    }}

    div[data-testid="stFileUploader"] button {{
        background: rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 8px;
    }}

    div[data-testid="stFileUploader"] button:hover {{
        background: rgba(255,255,255,0.25) !important;
    }}

    </style>
    """, unsafe_allow_html=True)

set_bg(img_path)

# =========================
# TITLE
# =========================
st.title("🧠 MNIST Autoencoder - Image Denoising")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# INPUT
# =========================
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
noise_factor = st.slider("Noise Level", 0.0, 1.0, 0.5)

# =========================
# MODEL
# =========================
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = Autoencoder().to(device)

    model.load_state_dict(
        torch.load(
            r"D:\Material of level 3\Semi 2\Applied ML\Denoising_Autoencoder\autoencoder.pth",
            map_location=device
        )
    )

    model.eval()
    return model

model = load_model()

# =========================
# HELPERS
# =========================
def preprocess(img):
    img = img.convert("L")
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def add_noise(img):
    noise = torch.randn_like(img) * noise_factor
    return torch.clamp(img + noise, 0., 1.)

# =========================
# RUN
# =========================
if uploaded_file:

    image = Image.open(uploaded_file)

    img_tensor = preprocess(image).to(device)
    noisy_img = add_noise(img_tensor)

    with torch.no_grad():
        output = model(noisy_img)

    mse = torch.mean((output - img_tensor) ** 2).item()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Original")
        st.image(img_tensor.squeeze().cpu().numpy(), width=150)

    with col2:
        st.markdown("### Noisy")
        st.image(noisy_img.squeeze().cpu().numpy(), width=150)

    with col3:
        st.markdown("### Reconstructed")
        st.image(output.squeeze().cpu().numpy(), width=150)

    st.markdown("---")
    st.metric("MSE Loss", round(mse, 6))