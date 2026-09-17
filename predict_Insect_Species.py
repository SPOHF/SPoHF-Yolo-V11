from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from ultralytics import YOLO
import tensorflow as tf
from PIL import ImageFont
import os
import time  # ← added

# Load the image
image_path = './Manual-Test-Data/Drosophila-melanogaster/SM01.jpg'
image = Image.open(image_path)
if image.mode in ('RGBA', 'LA', 'P'):
    image = image.convert('RGB')
image_np = np.array(image)

# Load the YOLO detection model
yolo_model = YOLO('./runs/detect/train9/weights/last.pt')

# Load the Keras classification model
classifier_model = tf.keras.models.load_model('./InsectClassificationModel/insect_classifier.keras')

# Get prediction parameters
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.010'))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.20'))

# ── Start timer ──────────────────────────────────────────────────────────────
start_time = time.perf_counter()

# Run YOLO inference to detect insects
results = yolo_model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
detected_insects = results[0].boxes
num_insects = len(detected_insects)
print(f"Number of insects detected: {num_insects}")

# Prepare image for drawing
draw_image = image.copy()
draw = ImageDraw.Draw(draw_image)

COLORS = {
    'Drosophila-Melanogaster': (255, 0, 0),
    'Others':   (0, 200, 0)
}
class_counts = {'Drosophila-Melanogaster': 0, 'Others': 0}

# Load a font
font_paths = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]
font = None
for path in font_paths:
    try:
        font = ImageFont.truetype(path, 20)
        break
    except:
        continue
if font is None:
    font = ImageFont.load_default()

# Process each detected insect
for idx, box in enumerate(detected_insects):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped_insect = image.crop((x1, y1, x2, y2))
    cropped_resized = cropped_insect.resize((224, 224))
    if cropped_resized.mode != 'RGB':
        cropped_resized = cropped_resized.convert('RGB')

    img_array = np.array(cropped_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = classifier_model.predict(img_array, verbose=0)[0][0]
    if prediction < 0.5:
        class_name = 'Drosophila-Melanogaster'
        confidence = (1 - prediction) * 100
    else:
        class_name = 'Others'
        confidence = prediction * 100

    class_counts[class_name] += 1
    color = COLORS[class_name]

    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    label = f"{class_name} {confidence:.1f}%"
    bbox = draw.textbbox((x1, y1), label, font=font)
    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
    draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    print(f"Insect {idx+1}: {class_name} ({confidence:.1f}%)")

# Show the final annotated image
draw_image.show()

# ── Stop timer & print ────────────────────────────────────────────────────────
elapsed_ms = (time.perf_counter() - start_time) * 1000
print(f"\n⏱  Detection + display took: {elapsed_ms:.2f} ms")

print(f"\nSummary:")
print(f"Total insects detected: {num_insects}")
print(f"Drosophila-Melanogaster: {class_counts['Drosophila-Melanogaster']}")
print(f"Others: {class_counts['Others']}")