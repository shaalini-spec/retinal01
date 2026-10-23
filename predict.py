
import json
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "outputs/best_model.keras"
CLASSES_PATH = "outputs/class_names.json"
IMG_SIZE = (224, 224)


def predict_image(image_path):
    print("Loading saved model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r") as file:
        class_names = json.load(file)

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    print("\n--- OCT Classification Result ---")
    print("Image:", image_path)
    print("Predicted class:", class_names[predicted_index])
    print(
        f"Model confidence: "
        f"{probabilities[predicted_index] * 100:.2f}%"
    )

    print("\nAll class probabilities:")
    for name, probability in sorted(
        zip(class_names, probabilities),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/oct_image.jpeg")
        sys.exit(1)

    predict_image(sys.argv[1])
