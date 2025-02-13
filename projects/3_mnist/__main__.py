import os
import tempfile

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

def classify_digit(model, image):
    img = cv2.imread(image)[:, :, 0]
    img = np.invert(np.array([img]))

    prediction = model.predict(img)
    os.remove(image)

    return prediction

def resize_image(path, target_size: (int, int)):
    image_np = np.array(Image.open(path))
    temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_image.png')
    cv2.imwrite(temp_image_path, image_np)

    img = Image.open(temp_image_path)
    resized_image = img.resize(target_size)

    return resized_image, temp_image_path

def predict_image(image_path: str):
    _, temp_path = resize_image(image_path, (300, 300))
    model = tf.keras.models.load_model("data/handwrittendigit.keras")

    return classify_digit(model, temp_path)

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=3)

    loss, accuracy = model.evaluate(X_test, y_test)

    print("Loss %:", loss)
    print("Accuracy %:", accuracy)

    model.save("data/handwrittendigit.keras")

    for i in range(8):
        if i in (3, 5):
            continue

        prediction = predict_image(f"data/mnist/{i}.png")

        print(f"Digit {i}: The digit is probably a {np.argmax(prediction)}.")
