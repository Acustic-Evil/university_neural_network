import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow import keras
from keras.models import model_from_json
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
import neural_training

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
    ar = ar.reshape(1, 50, 50, 3)
    prediction_score = model.predict(ar, verbose=1)
    label_index = np.argmax(prediction_score)
    acc = np.max(prediction_score)
    animal = get_animal_name(label_index)
    return animal, acc

def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    animal, acc = predict_animal(file_path)
    display_result(file_path, animal, acc)

def display_result(image_path, animal, acc):
    load = Image.open(image_path)
    render = ImageTk.PhotoImage(load)
    
    img = Label(image=render)
    img.image = render
    img.grid(column=1, row=1, padx=10, pady=10)

    result_text.set(f"The predicted Animal is a {animal} with accuracy = {acc}")
    result_label.grid(column=1, row=2, padx=10, pady=10)

def create_gui():
    global result_text, result_label

    root = tk.Tk()
    root.title("Animal Predictor")

    upload_btn = Button(root, text="Upload Image", command=upload_image)
    upload_btn.grid(column=0, row=0, padx=10, pady=10)

    result_text = tk.StringVar()
    result_label = Label(root, textvariable=result_text, font=("Helvetica", 16))

    root.mainloop()

if __name__ == '__main__':
    create_gui()
