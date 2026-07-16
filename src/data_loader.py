from sklearn.datasets import load_iris
import pandas as pd


def load_dataset():
    """
    Load the Iris dataset and return it as a pandas DataFrame.
    """

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["target"] = iris.target

    df["species"] = df["target"].map(
        {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }
    )

    return df