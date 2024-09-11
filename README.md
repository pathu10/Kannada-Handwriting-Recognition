# Kannada Handwritten Recognition

This project aims to recognize handwritten characters in the Kannada script using image processing techniques and machine learning.

## Project Structure

```
KannadaHandwrittenRecognition/
│
├── data/
│   ├── raw/                 # Raw images dataset
│   └── processed/           # Processed images storage
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Preprocessing functions
│   ├── feature_extraction.py# Feature extraction (HOG)
│   ├── model.py             # Model training and saving
│   ├── evaluate.py          # Model evaluation
│   └── gui.py               # Graphical User Interface
│
├── main.py                  # Main script for training and evaluation
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/pathu10/Kannada-Handwriting-Recognition.git
cd Kannada-Handwriting-Recognition
```

### 2. Install Dependencies

Ensure you have Python 3 and pip installed. Install required dependencies using:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place your raw handwritten Kannada character images in the `data/raw/` directory. Organize them into subfolders named after their respective labels (e.g., `0`, `1`, ..., `9`).

Example:

```
data/
└── raw/
    ├── 0/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    ├── 1/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    └── ...
```

### 4. Train the Model

Run the main script to train the model:

```bash
python main.py
```

This script will:
- Load and preprocess the images
- Save preprocessed images to `data/processed/`
- Extract HOG features
- Train an SVM model
- Evaluate the model's accuracy, precision, recall, F1-score, and support

### 5. Run the GUI for Prediction

To use the graphical interface for predicting characters:

```bash
python src/gui.py
```

This GUI allows you to load an image and predict the handwritten Kannada character using the trained model.

## Detailed Explanation of Scripts

### `main.py`
This script coordinates the entire process:
- Loads images from the `data/raw/` directory
- Preprocesses the images and saves them to `data/processed/`
- Extracts features using HOG (Histogram of Oriented Gradients)
- Trains an SVM model
- Evaluates the trained model and prints accuracy, precision, recall, F1-score, and classification report

### `src/preprocess.py`
Contains functions for:
- Loading images from the dataset
- Preprocessing images (e.g., resizing and thresholding)
- Saving preprocessed images to the `data/processed/` directory

### `src/feature_extraction.py`
Defines the function for extracting HOG features from the preprocessed images.

### `src/model.py`
Includes functions for training the SVM model and saving the trained model.

### `src/evaluate.py`
Contains functions for evaluating the model's performance. It calculates metrics such as precision, recall, F1-score, and support, and provides a classification report.

### `src/gui.py`
Provides a graphical user interface for loading images and predicting characters using the trained model.

## Metrics Explained

- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. It measures the accuracy of the positive predictions.
- **Recall**: The ratio of correctly predicted positive observations to all actual positives. It measures the ability of the model to identify positive instances.
- **F1-Score**: The weighted average of Precision and Recall. It balances the two metrics and is especially useful when dealing with imbalanced classes.
- **Support**: The number of actual occurrences of the class in the dataset. It represents the count of samples in each class.

## Example Classification Report

The classification report provides detailed performance metrics for each class, including:

```
              precision    recall  f1-score   support

           0       0.90      0.85      0.87       100
           1       0.80      0.75      0.77        80
           2       0.75      0.82      0.78        50

    accuracy                           0.82       230
   macro avg       0.81      0.81      0.81       230
weighted avg       0.83      0.82      0.82       230
```

- **Accuracy**: The overall accuracy of the model.
- **Macro Avg**: The average of precision, recall, and F1-score for each class, treating all classes equally.
- **Weighted Avg**: The average of precision, recall, and F1-score, weighted by the number of true instances for each class.

## Additional Notes

- Ensure you have a stable internet connection when installing dependencies, as some packages may require downloading additional files.
- Make sure your Python environment is correctly set up and all dependencies are installed before running the scripts.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

This README now includes a section on the metrics used in the classification report to help users understand the evaluation of the model's performance.
