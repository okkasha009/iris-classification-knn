from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """
    Splits the dataset into training and testing sets
    and applies feature scaling.
    """

    # Features (Input)
    X = df.drop(columns=["target", "species"])

    # Target (Output)
    y = df["target"]

    # Split into Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler