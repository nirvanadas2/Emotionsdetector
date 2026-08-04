import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def build_and_train_model():
    print("✅ Script started...")

    # Paths to training and testing directories
    train_dir = "train"
    test_dir = "test"

    # Verify dataset folders exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("❌ ERROR: 'train' or 'test' folder not found.")
        return

    # Image generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Generate batches of tensor image data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical"
    )

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical"
    )

    # Model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotion classes
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=25,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )

    # Save model architecture and weights
    model_json = model.to_json()
    with open("facialemotionmodel.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("facialemotionmodel.weights.h5")

    print("✅ Model saved successfully.")
    return model

if __name__ == "__main__":
    build_and_train_model()



