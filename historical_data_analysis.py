import os
os.environ["KERAS_BACKEND"] = "torch"   # MUSS vor dem keras-Import stehen

import keras
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pillow_heif import register_heif_opener
from concurrent.futures import ThreadPoolExecutor

# Register HEIF/HEIC support with Pillow
register_heif_opener()

# Load environment variables
load_dotenv()

# ============== CONFIG ==============

# Cores fuer das threaded EXIF-Prefetching (I/O-bound)
NUM_CORES = os.cpu_count()

historical_data_dir = "./historical_data"
results_dir = "./historical_data_results"
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, 'historical_insect_data.csv')

# Get prediction parameters from environment variables with defaults
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.20'))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.20'))

# Model paths (ueber .env ueberschreibbar)
YOLO_WEIGHTS = os.getenv('YOLO_WEIGHTS', './runs/detect/train-9/weights/best.pt')
CLASSIFIER_PATH = os.getenv('CLASSIFIER_PATH', './InsectClassificationModel/insect_classifier.keras')

# FORCE_REPROCESS=1 -> bestehende CSV ignorieren und alles neu rechnen
FORCE_REPROCESS = os.getenv('FORCE_REPROCESS', '0') == '1'

# Classification batch size (process multiple crops at once instead of one-by-one)
CLASSIFICATION_BATCH_SIZE = 32

# Kuerzel des verwendeten Modells, landet als Spalte in der CSV
MODEL_TAG = os.path.basename(os.path.dirname(os.path.dirname(YOLO_WEIGHTS))) + '/' + os.path.basename(YOLO_WEIGHTS)

# Spalten, die aus 'date' abgeleitet werden und nach jedem Merge neu berechnet werden
DERIVED_COLUMNS = ['year', 'week', 'calendar_week']


def extract_datetime_from_image(image_path):
    """Extract the date and time from EXIF metadata of an image (supports JPG, PNG, HEIC)."""
    try:
        image = Image.open(image_path)

        exif_data = None

        # Method 1: _getexif() works for JPG/JPEG
        if hasattr(image, '_getexif'):
            try:
                exif_data = image._getexif()
            except Exception:
                exif_data = None

        # Method 2: getexif() works for HEIC via pillow-heif and also JPG
        if exif_data is None:
            try:
                exif_info = image.getexif()
                if exif_info:
                    exif_data = dict(exif_info)
                    # EXIF sub-IFD (tag 0x8769) contains DateTimeOriginal for HEIC
                    exif_ifd = exif_info.get_ifd(0x8769)
                    if exif_ifd:
                        exif_data.update(exif_ifd)
            except Exception:
                exif_data = None

        if exif_data is not None:
            # First pass: look for DateTimeOriginal (preferred)
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "DateTimeOriginal":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")

            # Second pass: fallback to DateTime
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "DateTime":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")

    except Exception as e:
        print(f"Could not extract EXIF date from {image_path}: {e}")

    # Last resort: use file modification time
    mod_time = os.path.getmtime(image_path)
    print(f"Using file modification time for {os.path.basename(image_path)}")
    return datetime.fromtimestamp(mod_time)


def process_image(image_path):
    """Process a single image through YOLO detection and batched classification."""
    image = Image.open(image_path)

    # Convert to RGB if needed (HEIC, PNG with alpha, etc.)
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')

    # Some HEIC images open as RGB already but just in case
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Run YOLO inference
    results = yolo_model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    detected_insects = results[0].boxes

    # Initialize counters
    class_counts = {'Muscidae': 0, 'Others': 0}

    if len(detected_insects) == 0:
        return class_counts, 0

    # Collect all crops first, then classify in one batch
    crops = []
    for box in detected_insects:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cropped_insect = image.crop((x1, y1, x2, y2))
        cropped_resized = cropped_insect.resize((224, 224))

        if cropped_resized.mode != 'RGB':
            cropped_resized = cropped_resized.convert('RGB')

        img_array = np.array(cropped_resized, dtype=np.float32) / 255.0
        crops.append(img_array)

    # Batch classify all crops at once (much faster than one-by-one)
    crops_batch = np.array(crops, dtype=np.float32)

    # Process in chunks if there are many detections
    all_predictions = []
    for i in range(0, len(crops_batch), CLASSIFICATION_BATCH_SIZE):
        batch = crops_batch[i:i + CLASSIFICATION_BATCH_SIZE]
        predictions = classifier_model.predict(batch, verbose=0, batch_size=CLASSIFICATION_BATCH_SIZE)
        all_predictions.extend(np.asarray(predictions).flatten())

    # Count classes from batch predictions
    for prediction in all_predictions:
        if prediction < 0.5:
            class_counts['Muscidae'] += 1
        else:
            class_counts['Others'] += 1

    return class_counts, len(detected_insects)


def prefetch_exif(image_paths):
    """Pre-extract EXIF dates using multiple threads (I/O bound task)."""
    with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        dates = list(executor.map(extract_datetime_from_image, image_paths))
    return dates


def add_derived_columns(frame):
    """(Neu-)Berechnung von year / week / calendar_week aus der date-Spalte."""
    frame = frame.drop(columns=[c for c in DERIVED_COLUMNS if c in frame.columns])
    iso = frame['date'].dt.isocalendar()
    frame['year'] = iso.year.astype(int)
    frame['week'] = iso.week.astype(int)
    frame['calendar_week'] = frame['year'].astype(str) + '-KW' + frame['week'].astype(str).str.zfill(2)
    return frame


# ============== LOAD EXISTING RESULTS ==============

print("=" * 60)
print("Historical Data Analysis")
print("=" * 60)

existing_df = pd.DataFrame()
processed_files = set()

if os.path.exists(csv_path) and not FORCE_REPROCESS:
    try:
        existing_df = pd.read_csv(csv_path, parse_dates=['date'])
        processed_files = set(existing_df['filename'].astype(str))
        print(f"Bestehende CSV gefunden: {len(existing_df)} Zeilen bereits verarbeitet")
    except Exception as e:
        print(f"CSV konnte nicht gelesen werden ({e}) - es wird eine neue angelegt")
        existing_df = pd.DataFrame()
        processed_files = set()
elif FORCE_REPROCESS:
    print("FORCE_REPROCESS=1 - bestehende CSV wird ignoriert und alles neu berechnet")
else:
    print("Keine bestehende CSV - es wird eine neue angelegt")

# ============== FIND NEW IMAGES ==============

supported_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.heic')
image_files = [
    f for f in os.listdir(historical_data_dir)
    if f.lower().endswith(supported_extensions) and not f.startswith('.')
]

new_files = sorted(f for f in image_files if f not in processed_files)

print(f"Bilder im Ordner: {len(image_files)} | davon neu: {len(new_files)}")

if len(image_files) == 0 and existing_df.empty:
    print(f"No images found in {historical_data_dir}")
    raise SystemExit(0)

if not new_files and existing_df.empty:
    print("Nichts zu verarbeiten und keine bestehenden Daten.")
    raise SystemExit(0)

# ============== PROCESS NEW IMAGES ==============

results_data = []

if new_files:
    # Modelle erst laden, wenn tatsaechlich etwas zu rechnen ist
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(
            f"YOLO-Gewichte nicht gefunden: {YOLO_WEIGHTS}\n"
            f"Verfuegbare Runs: {os.listdir('./runs/detect') if os.path.isdir('./runs/detect') else 'kein runs/detect'}"
        )

    print(f"\nLade YOLO-Modell: {YOLO_WEIGHTS}")
    yolo_model = YOLO(YOLO_WEIGHTS)

    print(f"Lade Klassifikator: {CLASSIFIER_PATH}")
    classifier_model = keras.saving.load_model(CLASSIFIER_PATH)

    print(f"\nExtrahiere EXIF-Daten ({NUM_CORES} Threads)...")
    new_paths = [os.path.join(historical_data_dir, f) for f in new_files]
    capture_dates = prefetch_exif(new_paths)
    print("EXIF-Extraktion abgeschlossen\n")

    for i, filename in enumerate(new_files):
        image_path = os.path.join(historical_data_dir, filename)

        print(f"[{i+1}/{len(new_files)}] Processing: {filename}")

        capture_date = capture_dates[i]
        class_counts, total_count = process_image(image_path)

        results_data.append({
            'filename': filename,
            'date': capture_date,
            'total_insects': total_count,
            'muscidae': class_counts['Muscidae'],
            'others': class_counts['Others'],
            'model': MODEL_TAG,
        })

        print(f"  Date: {capture_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Total: {total_count} | Muscidae: {class_counts['Muscidae']} | Others: {class_counts['Others']}")
        print()
else:
    print("\nKeine neuen Bilder - Charts werden aus den bestehenden Daten neu erzeugt.\n")

# ============== MERGE AND SAVE ==============

new_df = pd.DataFrame(results_data)

if not existing_df.empty and not new_df.empty:
    df = pd.concat([existing_df, new_df], ignore_index=True)
elif not new_df.empty:
    df = new_df
else:
    df = existing_df.copy()

# Alte Zeilen bleiben erhalten; ein erneut verarbeitetes Bild ersetzt seinen alten Eintrag
df['date'] = pd.to_datetime(df['date'])
df = df.drop_duplicates(subset='filename', keep='last')
df = df.sort_values('date').reset_index(drop=True)
df = add_derived_columns(df)

if 'model' not in df.columns:
    df['model'] = ''
df['model'] = df['model'].fillna('')

# Spaltenreihenfolge stabil halten
column_order = ['filename', 'date', 'total_insects', 'muscidae', 'others', 'model'] + DERIVED_COLUMNS
df = df[[c for c in column_order if c in df.columns]]

df.to_csv(csv_path, index=False)
print(f"CSV aktualisiert: {csv_path}  ({len(new_df)} neu, {len(df)} gesamt)")

# ============== GENERATE CHARTS ==============

# Format date labels for x-axis
date_labels = df['date'].dt.strftime('%Y-%m-%d\n%H:%M')

# 1. Population over time (double bar chart per image)
fig, ax = plt.subplots(figsize=(max(12, len(df) * 1.5), 6))

x = np.arange(len(df))
bar_width = 0.35

bars_muscidae = ax.bar(x - bar_width/2, df['muscidae'], bar_width, label='Muscidae', color='#d32f2f')
bars_others = ax.bar(x + bar_width/2, df['others'], bar_width, label='Others', color='#388e3c')

ax.set_xlabel('Capture Date')
ax.set_ylabel('Number of Insects')
ax.set_title('Insect Population Over Time (per Yellow Card Image)')
ax.set_xticks(x)
ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars_muscidae:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

for bar in bars_others:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'population_over_time.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: population_over_time.png")

# 2. Total insect count over time (line chart per image)
fig, ax = plt.subplots(figsize=(max(12, len(df) * 1.5), 6))

ax.plot(x, df['total_insects'], marker='o', linewidth=2, color='#1565c0', label='Total Insects')
ax.plot(x, df['muscidae'], marker='s', linewidth=2, color='#d32f2f', label='Muscidae')
ax.plot(x, df['others'], marker='^', linewidth=2, color='#388e3c', label='Others')

ax.set_xlabel('Capture Date')
ax.set_ylabel('Number of Insects')
ax.set_title('Insect Population Trend Over Time')
ax.set_xticks(x)
ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'population_trend.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: population_trend.png")

# 3. Average population per calendar week (aggregated)
kw_grouped = df.groupby('calendar_week', sort=False).agg(
    muscidae_avg=('muscidae', 'mean'),
    others_avg=('others', 'mean'),
    total_avg=('total_insects', 'mean'),
    image_count=('filename', 'count'),
    year=('year', 'first'),
    week=('week', 'first')
).reset_index()

# Sort by year and week number
kw_grouped = kw_grouped.sort_values(['year', 'week']).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(max(12, len(kw_grouped) * 2), 6))

x_kw = np.arange(len(kw_grouped))
bar_width = 0.35

bars_m = ax.bar(x_kw - bar_width/2, kw_grouped['muscidae_avg'], bar_width, label='Muscidae (avg)', color='#d32f2f')
bars_o = ax.bar(x_kw + bar_width/2, kw_grouped['others_avg'], bar_width, label='Others (avg)', color='#388e3c')

ax.set_xlabel('Calendar week')
ax.set_ylabel('Average Number of Insects per Image')
ax.set_title('Average Insect Population per Calendar Week')
ax.set_xticks(x_kw)

# Label with KW and image count
kw_labels = [f"{row['calendar_week']}\n(n={int(row['image_count'])})" for _, row in kw_grouped.iterrows()]
ax.set_xticklabels(kw_labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars_m:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

for bar in bars_o:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'population_per_calendar_week.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: population_per_calendar_week.png")

# 4. Calendar week trend line
fig, ax = plt.subplots(figsize=(max(12, len(kw_grouped) * 2), 6))

ax.plot(x_kw, kw_grouped['total_avg'], marker='o', linewidth=2, color='#1565c0', label='Total (avg)')
ax.plot(x_kw, kw_grouped['muscidae_avg'], marker='s', linewidth=2, color='#d32f2f', label='Muscidae (avg)')
ax.plot(x_kw, kw_grouped['others_avg'], marker='^', linewidth=2, color='#388e3c', label='Others (avg)')

ax.set_xlabel('Calendar_Week')
ax.set_ylabel('Average Number of Insects per Image')
ax.set_title('Average Insect Population Trend per Calendar Week')
ax.set_xticks(x_kw)
ax.set_xticklabels(kw_labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'trend_per_calendar_week.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: trend_per_calendar_week.png")

# 5. Box plot per calendar week (min, max, median, mean, quartiles)
kw_order = kw_grouped['calendar_week'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(max(18, len(kw_grouped) * 3), 7))

for ax, column, title, color in [
    (axes[0], 'total_insects', 'Total Insects', '#1565c0'),
    (axes[1], 'muscidae', 'Muscidae', '#d32f2f'),
    (axes[2], 'others', 'Others', '#388e3c'),
]:
    # Group data per calendar week in correct order
    data_per_kw = [df[df['calendar_week'] == kw][column].values for kw in kw_order]

    bp = ax.boxplot(
        data_per_kw,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='gold', markeredgecolor='black', markersize=7),
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(facecolor=color, alpha=0.6),
        whiskerprops=dict(color=color, linewidth=1.5),
        capprops=dict(color=color, linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor=color, alpha=0.5),
    )

    # Add image count labels
    box_kw_labels = [f"{kw}\n(n={int(kw_grouped[kw_grouped['calendar_week'] == kw]['image_count'].values[0])})" for kw in kw_order]
    ax.set_xticks(np.arange(1, len(kw_order) + 1))
    ax.set_xticklabels(box_kw_labels, rotation=45, ha='right', fontsize=7)
    ax.set_title(title)
    ax.set_ylabel('Count per Image')
    ax.grid(axis='y', alpha=0.3)

# Add legend for mean marker
axes[0].plot([], [], marker='D', color='gold', markeredgecolor='black', linestyle='None', label='Mean')
axes[0].plot([], [], color='black', linewidth=2, label='Median')
axes[0].legend(loc='upper left', fontsize=8)

fig.suptitle('Insect Count Distribution per Calendar Week', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'boxplot_per_calendar_week.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: boxplot_per_calendar_week.png")

# 6. Summary pie chart
fig, ax = plt.subplots(figsize=(8, 6))

total_muscidae = df['muscidae'].sum()
total_others = df['others'].sum()

ax.pie(
    [total_muscidae, total_others],
    labels=['Muscidae', 'Others'],
    autopct='%1.1f%%',
    colors=['#d32f2f', '#388e3c'],
    startangle=90,
    textprops={'fontsize': 14}
)
ax.set_title(f'Overall Insect Distribution (Total: {total_muscidae + total_others})')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'distribution_pie_chart.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: distribution_pie_chart.png")

# ============== PRINT SUMMARY ==============

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Neu verarbeitet in diesem Lauf: {len(new_df)}")
print(f"Total images in dataset: {len(df)}")
print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Total insects detected: {df['total_insects'].sum()}")
print(f"Total Muscidae: {total_muscidae}")
print(f"Total Others: {total_others}")
print(f"Average insects per image: {df['total_insects'].mean():.1f}")
print(f"Peak count: {df['total_insects'].max()} ({df.loc[df['total_insects'].idxmax(), 'filename']})")

models_used = sorted(m for m in df['model'].unique() if m)
if len(models_used) > 1:
    print(f"\nWARNUNG: Zeilen stammen aus mehreren Modellen: {', '.join(models_used)}")
    print("Fuer konsistente Zahlen einmal mit FORCE_REPROCESS=1 durchlaufen lassen.")
elif models_used:
    print(f"Modell: {models_used[0]}")

print(f"\nPer Calendar weeks:")
for _, row in kw_grouped.iterrows():
    print(f"  {row['calendar_week']}: avg {row['total_avg']:.1f} insects ({int(row['image_count'])} images)")

print(f"\nAll results saved to: {results_dir}/")
print("=" * 60)