import os
import roboflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Roboflow with your API key
api_key = os.getenv('ROBOFLOW_API_KEY')
if not api_key:
    raise SystemExit("ROBOFLOW_API_KEY is not set - add it to your .env file")

rf = roboflow.Roboflow(api_key=api_key)

# Load the project
project = rf.workspace().project("spohf-kur4x-dokg9")

# Specify the dataset version ID
version = project.version('10')  # Replace VERSION_ID with the correct version ID

# Deploy the model weights
version.deploy("yolo26s", "/Users/christiansalz/Desktop/SPoHF-YoloV11/runs/detect/train-9/weights/", "best.pt")
