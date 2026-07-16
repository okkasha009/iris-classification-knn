from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"\nAccuracy : {accuracy * 100:.2f}%")

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    os.makedirs("screenshots", exist_ok=True)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=["Setosa","Versicolor","Virginica"],
        yticklabels=["Setosa","Versicolor","Virginica"]
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("screenshots/confusion_matrix.png")

    plt.show()

    print("\n✅ Confusion Matrix saved successfully.")