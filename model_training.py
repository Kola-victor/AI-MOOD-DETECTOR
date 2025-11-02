import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import os

# ============================================
# PATHS TO DATASETS
# ============================================
train_dir = 'datasets/train'
test_dir = 'datasets/test'

# ============================================
# DATA PREPROCESSING
# ============================================
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

# ============================================
# BUILDING THE CNN MODEL
# ============================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

# ============================================
# COMPILING THE MODEL
# ============================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# TRAINING THE MODEL
# ============================================
epochs = 25
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs
)

# ============================================
# SAVING THE MODEL
# ============================================
model.save('face_emotionModel.h5')
print("✅ Model training complete. Saved as face_emotionModel.h5")

# ============================================
# MODEL SUMMARY
# ============================================
model.summary()
model.save("face_expression_model.h5")
print("✅ Model saved successfully as face_expression_model.h5")