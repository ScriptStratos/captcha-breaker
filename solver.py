# solver.py

import cv2
import numpy as np
import tensorflow as tf
from loguru import logger

from model import create_model
# Refactored decoder - 2026-03-11
# Refactored preprocessor - 2026-03-11
# Refactored decoder - 2026-03-11

# Constants
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50
CHAR_SET = "0123456789abcdefghijklmnopqrstuvwxyz"
CHAR_SET_LENGTH = len(CHAR_SET)
NUM_CLASSES = CHAR_SET_LENGTH + 1  # +1 for the CTC blank label

# Load the pre-trained model
model = create_model(num_classes=NUM_CLASSES)
# In a real scenario, you would load trained weights
# model.load_weights("path/to/your/model_weights.h5")
logger.info("Model created (weights are not loaded in this demo).")


def preprocess_image(image_path):
    """Load and preprocess a single CAPTCHA image."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.expand_dims(img, axis=0)   # Add batch dimension
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

def decode_prediction(prediction):
    """Decode the raw model output into human-readable text."""
    # Use CTC greedy decoder
    input_len = np.ones(prediction.shape[0]) * prediction.shape[1]
    decoded_dense = tf.keras.backend.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0]
    
    # Convert numeric labels to characters
    decoded_text = ""
    for label in decoded_dense[0]:
        if label < CHAR_SET_LENGTH:
            decoded_text += CHAR_SET[label]
    return decoded_text

def solve_captcha(image_path):
    """Solve a CAPTCHA from an image file."""
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None

    # Get model prediction
    prediction = model.predict(processed_image)
    
    # Decode the prediction
    solution = decode_prediction(prediction)
    return solution


if __name__ == "__main__":
    # This is a placeholder for a real test.
    # You would need a sample CAPTCHA image to test this.
    logger.warning("This script requires a sample CAPTCHA image to run.")
    logger.info("Example usage: solution = solve_captcha(\"sample_captcha.png\")")
    # Create a dummy image for demonstration purposes
    dummy_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.uint8)
    cv2.imwrite("dummy_captcha.png", dummy_image)
    
    solution = solve_captcha("dummy_captcha.png")
    # The solution will be empty because the model is not trained
    print(f"Attempted to solve dummy_captcha.png. Solution: ", solution)
