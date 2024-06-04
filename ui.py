import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

def display_result(image_path, animal, acc, prediction_score):
    load = Image.open(image_path)
    load = load.resize((1280, 720), Image.LANCZOS)  
    render = ImageTk.PhotoImage(load)

    img_label.config(image=render)
    img_label.image = render

    result_text.set(f"The predicted Animal is a {animal} with accuracy = {acc}\nScores: {prediction_score}")
    result_label.grid(column=0, row=2, padx=10, pady=10)

def create_gui(upload_image_callback):
    global result_text, result_label, img_label

    root = tk.Tk()
    root.title("Animal Classifier")
    root.geometry("800x600")

    main_frame = tk.Frame(root)
    main_frame.pack(padx=20, pady=20)

    upload_btn = Button(main_frame, text="Upload Image", command=upload_image_callback, font=("Helvetica", 14))
    upload_btn.grid(column=0, row=0, padx=10, pady=10)

    img_label = Label(main_frame)
    img_label.grid(column=0, row=1, padx=10, pady=10)

    result_text = tk.StringVar()
    result_label = Label(main_frame, textvariable=result_text, font=("Helvetica", 16))
    result_label.grid(column=0, row=2, padx=10, pady=10)

    root.mainloop()
