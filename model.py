# model.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, Dense, LSTM, Input, Bidirectional
from tensorflow.keras.models import Model
# Refactored decoder - 2026-03-11
# Refactored trainer - 2026-03-11
# Refactored trainer - 2026-03-11
# Refactored model - 2026-03-11
# Refactored preprocessor - 2026-03-11
# Refactored model - 2026-03-11
# Refactored decoder - 2026-03-11
# Refactored solver - 2026-03-11
# Refactored model - 2026-03-11
# Refactored preprocessor - 2026-03-11
# Refactored model - 2026-03-11
# Refactored model - 2026-03-11
# Refactored solver - 2026-03-11
# Refactored solver - 2026-03-11
# Refactored model - 2026-03-11
# Refactored solver - 2026-03-11
# Refactored decoder - 2026-03-11
# Refactored solver - 2026-03-11
# Refactored decoder - 2026-03-11
# Refactored trainer - 2026-03-11
# Refactored model - 2026-03-11
# Refactored preprocessor - 2026-03-11
# Refactored model - 2026-03-11
# Refactored model - 2026-03-11
# Refactored trainer - 2026-03-11

# Constants
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50

def create_model(num_classes):
    """Create the CNN + RNN model for CAPTCHA solving."""
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="image_input", dtype="float32")

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    # Reshape for RNN layers
    # The output of the CNN is (batch_size, height, width, channels)
    # We need to reshape it to (batch_size, width, height * channels) for the RNN
    new_shape = (x.shape[2], x.shape[1] * x.shape[3])
    x = Reshape(target_shape=new_shape)(x)
    x = Dense(64, activation="relu")(x)

    # Recurrent layers (LSTM)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # Output layer
    output = Dense(num_classes, activation="softmax", name="output")(x)

    # Create the model
    model = Model(inputs=input_img, outputs=output, name="captcha_solver")

    # Define the CTC loss function
    def ctc_loss(y_true, y_pred):
        return tf.keras.backend.ctc_batch_cost(
            y_true,
            y_pred,
            input_length=tf.ones(tf.shape(y_pred)[0]) * y_pred.shape[1],
            label_length=tf.ones(tf.shape(y_pred)[0]) * tf.shape(y_true)[1],
        )

    # Compile the model
    model.compile(optimizer="adam", loss=ctc_loss, metrics=["accuracy"])

    return model


if __name__ == "__main__":
    # Example of creating the model
    from solver import NUM_CLASSES

    model = create_model(NUM_CLASSES)
    model.summary()
