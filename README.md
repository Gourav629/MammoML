# ThermalResNet — Thermal Image Classification

This repository contains code for training and evaluating a deep learning model to classify thermal medical images into Healthy and Sick categories using PyTorch and a fine-tuned ResNet18.

🚀 Features

Preprocessing pipeline with resizing, normalization, and dataset splitting.

Transfer learning with ResNet18 pretrained on ImageNet.

Training and validation loop with loss/ROC-AUC tracking.

Visualization of training curves, ROC-AUC, predictions, and confusion matrix.

Model saving and loading for deployment.

⚙️ Installation

Clone the repo and install dependencies:

git clone https://github.com/yourusername/ThermalResNet.git
cd ThermalResNet

# Create environment
conda create -n thermal python=3.9
conda activate thermal

# Install requirements
pip install torch torchvision scikit-learn matplotlib seaborn

# Model Architecture 
link : https://gist.github.com/user-attachments/assets/e2664ff8-2752-4f67-9cd6-40f9aba3a4d4

🏃 Usage
1. Train Model
python train.py

2. Evaluate Model
python evaluate.py

3. Inference on New Images
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load("mammo_ml.pth")
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open("sample.png")
input_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    _, pred = torch.max(output, 1)
    print("Prediction:", "Healthy" if pred.item() == 0 else "Sick")

📊 Results

Validation Accuracy: ~94.6%

Precision (Sick): ~100%

Recall (Sick): ~90%

F1-Score: ~0.95

High ROC-AUC across epochs (>0.99 at peak).

Visual outputs include:

Training & validation loss curves

Validation ROC-AUC curve

Sample predictions

Confusion matrix

🛠️ Files

train.py → Training loop with ResNet18.

evaluate.py → Model evaluation and confusion matrix.

mammo_ml.pth → Saved trained model.

utils.py → Helper functions for visualization.

📌 Notes

ResNet weights are automatically downloaded from PyTorch Hub.

Modify data_dir in code to point to your dataset folder.

Works with both CPU and CUDA.
