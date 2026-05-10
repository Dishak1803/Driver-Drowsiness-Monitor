# train_yawn.py
# PURPOSE : Train a CNN model to detect Yawn vs No Yawn
# INPUT   : dataset1/  →  yawn/  and  no yawn/
# OUTPUT  : yawn_model.h5  (saved trained model)
# ─────────────────────────────────────────────────────────────

import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
DATASET1   = r"C:\Users\Disha\Downloads\dataset1"        # yawn / no yawn
MODEL_SAVE = r"C:\Users\Disha\Driver_Drowsiness\yawn_model.h5"

# ─────────────────────────────────────────────────────────────
# STEP 1 — Prepare Data Generators
#   ImageDataGenerator handles:
#     - rescale     : converts pixel values from 0-255 → 0-1
#     - augmentation: randomly flips/zooms/rotates images
#                     so the model doesn't overfit
#     - validation_split: automatically splits 80% train / 20% val
# ─────────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1./255,           # normalize pixel values
    rotation_range=10,        # randomly rotate images by up to 10°
    zoom_range=0.1,           # randomly zoom in by up to 10%
    horizontal_flip=True,     # randomly flip images left/right
    validation_split=0.2      # reserve 20% of images for validation
)

# Training set — 80% of dataset1
train_data = datagen.flow_from_directory(
    DATASET1,
    target_size=(64, 64),     # resize all images to 64x64 pixels
    batch_size=32,            # process 32 images at a time
    class_mode='binary',      # binary = 2 classes (yawn / no yawn)
    subset='training',
    shuffle=True
)

# Validation set — 20% of dataset1
val_data = datagen.flow_from_directory(
    DATASET1,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Print class mapping so we know which label = which number
print("\n📌 Class Indices:", train_data.class_indices)
# Expected → {'no yawn': 0, 'yawn': 1}

print(f"   Training   batches : {len(train_data)}")
print(f"   Validation batches : {len(val_data)}\n")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Build the CNN Model
#   Architecture:
#     3x [Conv2D → MaxPooling]  : extract features from images
#     Flatten                   : convert 2D features to 1D
#     Dense(128)                : learn patterns from features
#     Dropout(0.5)              : randomly turn off 50% neurons
#                                 during training → prevents overfitting
#     Dense(1, sigmoid)         : output single value 0.0 - 1.0
#                                 <0.5 = No Yawn, >0.5 = Yawn
# ─────────────────────────────────────────────────────────────
model = models.Sequential([

    # Block 1 — detect basic edges and shapes
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    # Block 2 — detect more complex patterns
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    # Block 3 — detect high-level features (open mouth, teeth etc.)
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    # Flatten 2D feature maps into a 1D vector
    layers.Flatten(),

    # Fully connected layer to learn combinations of features
    layers.Dense(128, activation='relu'),

    # Dropout — prevents overfitting by randomly dropping neurons
    layers.Dropout(0.5),

    # Output layer — 1 neuron, sigmoid gives value between 0 and 1
    layers.Dense(1, activation='sigmoid')
])

# ─────────────────────────────────────────────────────────────
# STEP 3 — Compile the Model
#   optimizer : adam adjusts learning rate automatically
#   loss      : binary_crossentropy is standard for 2-class problems
#   metrics   : we track accuracy during training
# ─────────────────────────────────────────────────────────────
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────────────────────
# STEP 4 — Callbacks
#   EarlyStopping  : stops training if val_loss doesn't improve
#                    for 5 epochs → saves time, prevents overfitting
#   ModelCheckpoint: saves the BEST version of the model automatically
#                    (best = lowest validation loss)
# ─────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_loss',   # watch validation loss
        patience=5,           # stop if no improvement for 5 epochs
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE,           # where to save the model
        monitor='val_loss',
        save_best_only=True,  # only save when model improves
        verbose=1
    )
]

# ─────────────────────────────────────────────────────────────
# STEP 5 — Train the Model
# ─────────────────────────────────────────────────────────────
print("\n🚀 Starting Yawn Model Training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,                # max 20 epochs (EarlyStopping may stop earlier)
    callbacks=callbacks
)

print(f"\n✅ Yawn model saved to: {MODEL_SAVE}")

# ─────────────────────────────────────────────────────────────
# STEP 6 — Plot Training Results
#   Accuracy plot : shows how well model learns over epochs
#   Loss plot     : shows if model is overfitting
#                   (if train loss drops but val loss rises = overfit)
# ─────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Yawn Model — Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Yawn Model — Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\Disha\Driver_Drowsiness\yawn_training_plot.png")
plt.show()
print("📊 Training plot saved!")