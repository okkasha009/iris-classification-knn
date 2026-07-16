from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def compare_models(X_train, X_test, y_train, y_test):

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200)
    }

    results = {}

    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON")
    print("=" * 60)

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        results[name] = accuracy

        print(f"{name:<22}: {accuracy*100:.2f}%")

    best_model = max(results, key=results.get)

    print("\n" + "-" * 60)
    print(f"🏆 Best Model : {best_model}")
    print("-" * 60)