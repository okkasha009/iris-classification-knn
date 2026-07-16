import joblib
import pandas as pd

species_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


def predict_flower(scaler):

    model = joblib.load("models/knn_model.pkl")

    print("\n" + "=" * 50)
    print("🌸 Flower Prediction")
    print("=" * 50)

    try:
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))

    except ValueError:
        print("\n❌ Please enter valid numeric values.")
        return

    sample = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]

    probabilities = model.predict_proba(sample_scaled)[0]

    confidence = max(probabilities) * 100

    print("\n" + "-" * 50)
    print(f"🌼 Predicted Species : {species_names[prediction]}")
    print(f"🎯 Confidence Score : {confidence:.2f}%")
    print("-" * 50)