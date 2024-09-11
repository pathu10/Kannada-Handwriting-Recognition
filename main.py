from src.preprocess import load_images, preprocess_images
from src.feature_extraction import extract_hog_features
from src.model import train_model
from src.evaluate import evaluate_model

def main():
    data_path = 'data/raw/'
    save_path = 'data/processed/'  # Path to save preprocessed images
    images, labels = load_images(data_path)
    preprocessed_images = preprocess_images(images, save_path)
    features = extract_hog_features(preprocessed_images)
    model, X_test, y_test = train_model(features, labels)
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print('Classification Report:')
    print(report)

if __name__ == "__main__":
    main()
