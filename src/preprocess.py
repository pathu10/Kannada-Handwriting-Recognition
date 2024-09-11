import cv2
import os
import numpy as np

def load_images(data_path):
    images = []
    labels = []
    for label in os.listdir(data_path):
        for img_name in os.listdir(os.path.join(data_path, label)):
            img_path = os.path.join(data_path, label, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32))  # Resize to a uniform size
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_image(image):
    binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
    return binary_image

def preprocess_images(images, save_path):
    preprocessed_images = []
    for i, img in enumerate(images):
        preprocessed_img = preprocess_image(img)
        preprocessed_images.append(preprocessed_img)
        save_preprocessed_image(preprocessed_img, i, save_path)
    return np.array(preprocessed_images)

def save_preprocessed_image(image, index, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_path = os.path.join(save_path, f'img_{index}.png')
    cv2.imwrite(save_img_path, image)
