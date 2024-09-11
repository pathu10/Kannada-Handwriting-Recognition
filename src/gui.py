import sys
import os
import cv2
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import preprocess_image
from feature_extraction import extract_hog_features

# Load the trained model
def load_model(model_path='model.pkl'):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        messagebox.showerror("Error", "Model file not found. Make sure 'model.pkl' exists in the project directory.")
        sys.exit(1)

# Predict the character from the image
def predict(image_path, model):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))
    image = preprocess_image(image)
    features = extract_hog_features([image])
    prediction = model.predict(features)
    return prediction[0]

# Open file dialog to select image
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        show_image(file_path)
        prediction = predict(file_path, model)
        result_label.config(text=f'Prediction: {prediction}')

# Display selected image in the GUI
def show_image(image_path):
    img = Image.open(image_path)
    img = img.resize((200, 200), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Clear the displayed image and result
def clear_results():
    image_label.config(image='')
    image_label.image = None
    result_label.config(text="Prediction: ")

# Load the model
model = load_model()

# Initialize the main window
root = tk.Tk()
root.title("Kannada Handwritten Recognition")
root.geometry("400x400")
root.resizable(False, False)

# Style configuration for better UI
style = ttk.Style(root)
style.configure('TButton', font=('Helvetica', 10))
style.configure('TLabel', font=('Helvetica', 12))

# Widgets
open_button = ttk.Button(root, text="Open Image", command=open_file_dialog)
open_button.pack(pady=10)

image_label = ttk.Label(root)
image_label.pack(pady=10)

result_label = ttk.Label(root, text="Prediction: ")
result_label.pack(pady=10)

clear_button = ttk.Button(root, text="Clear", command=clear_results)
clear_button.pack(pady=10)

# Run the application
root.mainloop()


