import os
import shutil

# ─────────────────────────────────────────────────────────────
# DATASET PATHS — relative to this script's location
#   Place your downloaded Kaggle dataset folders here before running:
#     data/dataset2/
#     data/dataset3/
# ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET2    = os.path.join(BASE_DIR, "data", "dataset2")
DATASET3    = os.path.join(BASE_DIR, "data", "dataset3")

# This is the NEW folder we will create to hold all merged eye images
EYES_MERGED = os.path.join(BASE_DIR, "data", "eyes_combined")

# ─────────────────────────────────────────────────────────────
# STEP 1 — Create the output folder structure
#   data/eyes_combined/
#       ├── open/
#       └── closed/
# ─────────────────────────────────────────────────────────────
os.makedirs(os.path.join(EYES_MERGED, 'open'),   exist_ok=True)
os.makedirs(os.path.join(EYES_MERGED, 'closed'), exist_ok=True)
print("✅ Output folders created at:", EYES_MERGED)

# ─────────────────────────────────────────────────────────────
# STEP 2 — Helper function to copy images
#   src_folder : where to copy FROM
#   label      : 'open' or 'closed' — which output folder to copy INTO
#   counter    : dict to keep track of how many files copied so far
#                (used to give each file a unique name → avoids overwriting)
# ─────────────────────────────────────────────────────────────
def copy_images(src_folder, label, counter):
    # Check the source folder actually exists before trying to copy
    if not os.path.exists(src_folder):
        print(f"  ⚠ NOT FOUND, skipping: {src_folder}")
        return

    # Destination folder inside eyes_combined
    dst_folder = os.path.join(EYES_MERGED, label)

    # Loop through every file in the source folder
    for img_name in os.listdir(src_folder):

        # Only copy image files — skip anything else (e.g. .txt, .xml)
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Give each file a unique name using the counter
        # e.g. img_00001.jpg, img_00002.jpg ...
        new_name = f"img_{counter[label]:05d}.jpg"

        # Do the actual file copy
        shutil.copy(
            os.path.join(src_folder, img_name),   # source file
            os.path.join(dst_folder, new_name)     # destination file
        )

        # Increase the counter for this label
        counter[label] += 1

# ─────────────────────────────────────────────────────────────
# STEP 3 — Start counting from 0 for both labels
# ─────────────────────────────────────────────────────────────
counter = {'open': 0, 'closed': 0}

# ─────────────────────────────────────────────────────────────
# STEP 4 — Copy from Dataset 2
#   Note: folder names have spaces → "close eyes" / "open eyes"
#         and images are inside data/train/ and data/test/
# ─────────────────────────────────────────────────────────────
print("\n📂 Copying from Dataset 2 (train)...")
copy_images(os.path.join(DATASET2, 'data', 'train', 'open eyes'),   'open',   counter)
copy_images(os.path.join(DATASET2, 'data', 'train', 'close eyes'),  'closed', counter)

print("📂 Copying from Dataset 2 (test)...")
copy_images(os.path.join(DATASET2, 'data', 'test', 'open eyes'),    'open',   counter)
copy_images(os.path.join(DATASET2, 'data', 'test', 'close eyes'),   'closed', counter)

# ─────────────────────────────────────────────────────────────
# STEP 5 — Copy from Dataset 3
#   Note: folder names use underscores → "Closed_Eyes" / "Open_Eyes"
# ─────────────────────────────────────────────────────────────
print("📂 Copying from Dataset 3 (train)...")
copy_images(os.path.join(DATASET3, 'train', 'Open_Eyes'),   'open',   counter)
copy_images(os.path.join(DATASET3, 'train', 'Closed_Eyes'), 'closed', counter)

# ─────────────────────────────────────────────────────────────
# STEP 6 — Print final summary so we can verify everything merged
# ─────────────────────────────────────────────────────────────
print("\n✅ MERGE COMPLETE!")
print(f"   Total OPEN   images : {counter['open']}")
print(f"   Total CLOSED images : {counter['closed']}")
print(f"   Grand Total         : {counter['open'] + counter['closed']}")
print(f"\n📁 Saved to: {EYES_MERGED}")
