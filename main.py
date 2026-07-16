from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.model import train_model, save_model, save_scaler
from src.evaluation import evaluate_model
from src.predictor import predict_flower
from src.visualization import create_visualizations
from src.model_comparison import compare_models

def main():

    print("=" * 60)
    print("        🌸 Iris Flower Classification System 🌸")
    print("=" * 60)

    # =====================================
    # Step 1: Load Dataset
    # =====================================
    df = load_dataset()
    print("\n✅ Dataset Loaded Successfully")

    # =====================================
    # Step 2: Create Dataset Visualizations
    # =====================================
    create_visualizations(df)
    print("✅ Dataset Visualizations Created")

    # =====================================
    # Step 3: Preprocess Dataset
    # =====================================
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print("✅ Data Preprocessed")

    # =====================================
    # Step 4: Train Model
    # =====================================
    model = train_model(X_train, y_train)
    print("✅ KNN Model Trained")

    # =====================================
    # Step 5: Save Model
    # =====================================
    save_model(model)

    # =====================================
    # Step 6: Save Scaler
    # =====================================
    save_scaler(scaler)

    # =====================================
    # Step 7: Evaluate Model
    # =====================================
    evaluate_model(model, X_test, y_test)
    # =====================================
    # Step 8: Compare ML Models
    # =====================================
    
    compare_models(
    X_train,
    X_test,
    y_train,
    y_test
)

    # =====================================
    # Step 9: Predict New Flower
    # =====================================
    predict_flower(scaler)

    print("\n" + "=" * 60)
    print("🎉 Project Executed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()