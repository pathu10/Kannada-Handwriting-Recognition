from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    return model, X_test, y_test


