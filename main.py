import os
import cv2
from PIL import Image
import numpy as np

from keras.models import model_from_json

from neural_training import neural_training


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
    image = img.resize((50, 50))
    return np.array(image)


def get_animal_name(label):
    if label == 0:
        return "deer"
    if label == 1:
        return "cat"
    if label == 2:
        return "dog"


def predict_animal(file):
    # Check if model files exist, if not, start training
    if not check_model_files():
        print("Model files not found, starting training...")
        neural_training()

    print("Predicting .................................")
    model = load_model()
    ar = convert_to_array(file)
    ar = ar / 255
    label = 1
    a = []
    a.append(ar)
    a = np.array(a)
    prediction_score = model.predict(a, verbose=1)
    print(prediction_score)
    label_index = np.argmax(prediction_score)
    print(label_index)
    acc = np.max(prediction_score)
    animal = get_animal_name(label_index)
    print(animal)
    print("The predicted Animal is a " + animal + " with accuracy =    " + str(acc))


if __name__ == '__main__':
    predict_animal('./deer/0bdd8b2ba3.jpg')
    predict_animal('./dogs/dog.4001.jpg')
