from PIL import Image
import numpy as np
import cv2  # Required for color conversion
from ultralytics import YOLO
import os
import uuid  # For generating unique random strings

# Load the image and resize it to 1280x1280
image_path = './Manual-Test-Data/3.jpg'
image = Image.open(image_path)

# Load the trained model
model = YOLO('./runs/detect/train9/weights/last.pt')

# Run inference on the image
results = model.predict(image, conf=0.10, iou=0.2)

# Display results with custom settings
annotated_image = results[0].plot(labels=True, font_size=4, line_width=1)

# Convert BGR to RGB
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Convert the result to a PIL Image
annotated_image_pil = Image.fromarray(annotated_image_rgb)

# Show the annotated image
annotated_image_pil.show()

detected_insects = results[0].boxes  # The 'boxes' attribute contains the bounding box info
num_insects = len(detected_insects)  # Count how many boxes were detected

print(f"Number of insects detected: {num_insects}")

# Create the directory for saving found insects if it doesn't exist
save_dir = './foundInsects'
os.makedirs(save_dir, exist_ok=True)

# Process each detected insect
for idx, box in enumerate(detected_insects):
    # Get the coordinates of the bounding box
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
    
    # Crop the insect from the original image
    cropped_insect = image.crop((x1, y1, x2, y2))
    
    # Generate a unique random string (9 characters)
    unique_id = uuid.uuid4().hex[:9]
    
    # Save the cropped insect with unique filename
    insect_filename = f"{save_dir}/insect_{unique_id}.jpg"
    cropped_insect.save(insect_filename)
    print(f"Saved insect to {insect_filename}")