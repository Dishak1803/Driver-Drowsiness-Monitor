# ─────────────────────────────────────────────────────────────
# train_eye.py
# PURPOSE : Train CNN to detect Open vs Closed Eyes
# INPUT   : dataset3/train/ → Closed_Eyes/ and Open_Eyes/
# OUTPUT  : eye_model.h5
# Using dataset3 only (4,000 images) — fast training on CPU
# ─────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
DATASET3   = r"C:\Users\Disha\Downloads\dataset3\train"  # Closed_Eyes / Open_Eyes
MODEL_SAVE = r"C:\Users\Disha\Driver_Drowsiness\eye_model.h5"

# ─────────────────────────────────────────────────────────────
# STEP 1 — Data Generator
#   dataset3 has no train/test split so we create one here
#   80% train, 20% validation using validation_split
# ─────────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1./255,           # normalize pixels 0-255 → 0-1
    rotation_range=10,        # random rotation up to 10°
    zoom_range=0.1,           # random zoom up to 10%
    horizontal_flip=True,     # random left/right flip
    validation_split=0.2      # 80% train / 20% val
)

# Training set
train_data = datagen.flow_from_directory(
    DATASET3,
    target_size=(64, 64),     # 64x64 is fine — only 4000 images
    batch_size=32,
    class_mode='binary',      # 2 classes → open / closed
    subset='training',
    shuffle=True
)

# Validation set
val_data = datagen.flow_from_directory(
    DATASET3,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# IMPORTANT — save this mapping, we need it in the webcam script
print("\n📌 Class Indices:", train_data.class_indices)
# Expected → {'Closed_Eyes': 0, 'Open_Eyes': 1}

print(f"   Training   images : {train_data.samples}")
print(f"   Validation images : {val_data.samples}\n")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Build CNN Model
# ─────────────────────────────────────────────────────────────
model = models.Sequential([

    # Block 1 — detect basic edges (eyelids, lashes)
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    # Block 2 — detect iris and pupil shapes
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    # Block 3 — detect open/closed eye patterns
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    # Flatten into 1D vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(128, activation='relu'),

    # Dropout — prevents overfitting
    layers.Dropout(0.5),

    # Output — 0=Closed_Eyes, 1=Open_Eyes
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────────────────────
# STEP 3 — Callbacks
# ─────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,               # stop if no improvement for 5 epochs
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE,
        monitor='val_loss',
        save_best_only=True,      # only save when model improves
        verbose=1
    )
]

# ─────────────────────────────────────────────────────────────
# STEP 4 — Train
#   Estimated time: 5-10 minutes on CPU ✅
# ─────────────────────────────────────────────────────────────
print("\n🚀 Starting Eye Model Training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=callbacks
)

print(f"\n✅ Eye model saved to: {MODEL_SAVE}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Plot Training Results
# ─────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Eye Model — Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Eye Model — Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\Disha\Driver_Drowsiness\eye_training_plot.png")
plt.show()
print("📊 Training plot saved!")
