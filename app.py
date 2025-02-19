import torch
import torch.nn as nn
import torchvision.transforms as transforms
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Define the same model architecture used for training


class FaceMaskCNN(nn.Module):
    def __init__(self):
        super(FaceMaskCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 2)  # Masked, Unmasked

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load model
model = FaceMaskCNN()
model.load_state_dict(torch.load("facemask_model.pth",
                      map_location=torch.device('cpu')))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Function to predict mask status


def predict_mask(image):
    img = Image.open(image).convert("RGB")  # Ensure image is in RGB
    img = transform(img)  # Apply transformations
    img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return "Masked 😷" if predicted.item() == 0 else "No Mask ❌"


# Streamlit UI
st.title("Face Mask Detection with CNN 🖥️😷")

st.write("Upload an image to check if a person is wearing a mask.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    result = predict_mask(uploaded_file)
    st.write("### Prediction:", result)
