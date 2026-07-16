import os
import matplotlib.pyplot as plt
import seaborn as sns


def create_visualizations(df):

    os.makedirs("screenshots", exist_ok=True)

    # ----------------------------
    # Pair Plot
    # ----------------------------

    pair = sns.pairplot(
        df,
        hue="species",
        palette="Set2"
    )

    pair.savefig("screenshots/pairplot.png")

    plt.close()

    # ----------------------------
    # Correlation Heatmap
    # ----------------------------

    plt.figure(figsize=(8,6))

    sns.heatmap(
        df.drop(columns=["species"]).corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig("screenshots/correlation_heatmap.png")

    plt.close()

    # ----------------------------
    # Species Distribution
    # ----------------------------

    plt.figure(figsize=(6,5))

    sns.countplot(
        data=df,
        x="species",
        palette="Set2"
    )

    plt.title("Species Distribution")

    plt.tight_layout()

    plt.savefig("screenshots/species_distribution.png")

    plt.close()

    print("\n✅ Dataset visualizations saved in screenshots/")