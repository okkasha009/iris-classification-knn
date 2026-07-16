from sklearn.neighbors import KNeighborsClassifier
import joblib
import os


def train_model(X_train, y_train, n_neighbors=5):
    """
    Train a KNN model.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def save_model(model, filename="models/knn_model.pkl"):
    """
    Save the trained model.
    """
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, filename)

    print(f"\n✅ Model saved successfully at: {filename}")


def save_scaler(scaler, filename="models/scaler.pkl"):
    """
    Save the trained scaler.
    """
    os.makedirs("models", exist_ok=True)

    joblib.dump(scaler, filename)

    print(f"✅ Scaler saved successfully at: {filename}")