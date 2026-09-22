from ultralytics import YOLO
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables with fallback-defaults
model_path = os.getenv('MODEL_PATH', 'yolo26s.pt')
data_path = os.getenv('DATA_PATH', './data.yaml')
epochs = int(os.getenv('EPOCHS', '150'))
device = os.getenv('DEVICE', 'mps')
imgsz = int(os.getenv('IMAGE_SIZE', '1024'))
patience = int(os.getenv('PATIENCE', '200'))

# Load the YOLO model
model = YOLO(model_path)

# Train the model
print(f"Training with: epochs={epochs}, device={device}, imgsz={imgsz}, patience={patience}")
model.train(data=data_path, epochs=epochs, device=device, imgsz=imgsz, patience=patience, deterministic=False)