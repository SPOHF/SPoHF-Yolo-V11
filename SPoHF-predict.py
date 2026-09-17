from PIL import Image
import numpy as np
import cv2  # Required for color conversion
from ultralytics import YOLO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get prediction parameters from environment variables with defaults
conf_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.70'))
iou_threshold = float(os.getenv('IOU_THRESHOLD', '0.20'))

# Load the image and resize it to:
image_path = './Manual-Test-Data/34.HEIC'
image = Image.open(image_path)
#image_resized = image.resize((640, 640))

# Load the trained model
# There are two versions after each run you can choose between last.pt and best.pt
model = YOLO('./runs/detect/train-9/weights/best.pt')

# Run inference on the resized image
# conf=0.40 the minimum confidence for diplaying boundingboxes
# iou=0.20 threshold value for the intersection over union, if you have cludded boxes (20% means if two boxes overlap by 20% only one will be drawn)
# Run YOLO inference to detect insects with env variables
results = model.predict(image, conf=conf_threshold, iou=iou_threshold)

# Display results with custom settings
annotated_image = results[0].plot(labels=True, font_size=6, line_width=2)

# Convert BGR to RGB
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Convert the result to a PIL Image
annotated_image_pil = Image.fromarray(annotated_image_rgb)

# Save or show the annotated image
annotated_image_pil.show()

detected_insects = results[0].boxes  # The 'boxes' attribute contains the bounding box info
num_insects = len(detected_insects)  # Count how many boxes (Insects) were detected

print(f"Number of insects detected: {num_insects}")