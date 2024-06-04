import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow import keras
from keras.models import model_from_json
import neural_training
from ui import create_gui, display_result

# Function to check if model files exist
def check_model_files():
    return os.path.exists('model.json') and os.path.exists('model.h5')


# Function to load the model from the JSON and H5 files
def load_model():
    # Load the model architecture from the JSON file
    with open('model.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)

    # Load the weights into the model
    model.load_weights('model.h5')
    return model


def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((128, 128))
    return np.array(image)


def get_animal_name(label):
    if label == 0:
        return "deer"
    if label == 1:
        return "cow"
    if label == 2:
        return "moose"
    if label == 3:
        return "unknown"


def predict_animal(file, threshold=0.5):
    if not check_model_files():
        print("Model files not found, starting training...")
        neural_training.neural_training()

    print("Predicting .................................")
    model = load_model()
    ar = convert_to_array(file)
    ar = ar / 255
    ar = ar.reshape(1, 128, 128, 3)
    prediction_score = model.predict(ar, verbose=1)
    label_index = np.argmax(prediction_score)
    acc = np.max(prediction_score)

    if acc < threshold or label_index == 3:
        print("Unknown object detected with confidence:", acc)
        raise ValueError(f"The model could not recognize the object. Scores: {prediction_score}")

    animal = get_animal_name(label_index)
    return animal, acc, prediction_score


def upload_image():
    from tkinter import filedialog, messagebox
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    try:
        animal, acc, prediction_score = predict_animal(file_path)
        display_result(file_path, animal, acc, prediction_score)
    except ValueError as e:
        messagebox.showerror("Error", str(e))


if __name__ == '__main__':
    create_gui(upload_image)
