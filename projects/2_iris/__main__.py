# Undersøg datasættet:
# - Er der fejl i datasættet?
# - Er der outliers?
# - Er datasættet korrekt formateret?
# - Er der eventuelt en ny feature der kan blive lavet ud fra det nuværende
#   data?
# - Undersøg de forskellige korrelation i mellem de forskellige features.
#   (Hint: pandas)
#
# Visualisering:
# - Lav et histrogram, der visualiser de forskellige features.
# - Lav et scatterplot der viser korrelation imellem de features med højest
#   score, og dem med lavest score.
# - Samlign de forskellige arter af blomster ved boxplot.

from sklearn.datasets import load_iris
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def show_histogram(df: pd.DataFrame):
    plt.hist(df["sepal area (cm)"], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Area (cm)")
    plt.ylabel("Amount")
    plt.show()

def show_scatterplot(df: pd.DataFrame):
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    X = df.iloc[0:100, [0, 2]].values
    plt.scatter(X[:50, 0], X[:50, 1],
                color='blue', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='green', marker='s', label='Versicolor')
    plt.xlabel('sepal area (cm)')
    plt.ylabel('petal area (cm)')
    plt.legend(loc='upper left')
    plt.show()

def show_boxplot(df: pd.DataFrame):
    plt.boxplot(df)
    plt.show()

def main():
    # Load the Iris dataset.
    iris = load_iris()

    df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]],
                      columns=iris["feature_names"] + ["target"])
    df["sepal area (cm)"] = df["sepal length (cm)"] * df["sepal width (cm)"]
    df["petal area (cm)"] = df["petal length (cm)"] * df["petal width (cm)"]

    show_scatterplot(df)

if __name__ == "__main__":
    main()
