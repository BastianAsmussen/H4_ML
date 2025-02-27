import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

def get_dataset(path: str) -> DataFrame:
    df = pd.read_csv(path)

    # Fix missing values.
    df.dropna(subset=["title", "type", "duration"], inplace=True)
    df.loc[:, "country"] = df["country"].fillna("Unknown")
    df.loc[:, "listed_in"] = df["listed_in"].fillna("Unknown")

    df["duration"] = df["duration"].apply(lambda x: x.split(" ")[0] if "min" in x else None)
    df.dropna(subset=["duration"], inplace=True)
    df["duration"] = df["duration"].astype(int)

    return df

def predict_length(df: DataFrame, genre: str) -> int:
    X = df[["duration"]]
    y = df["listed_in"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=X.columns)
    plt.title("Feature Importance")
    plt.show()

    return y_pred

def analyze_hypotheses(df: DataFrame):
    # Hypothesis 1: Films have become longer over the years.
    movies = df.copy()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=movies, x="release_year", y="duration", errorbar=None)
    plt.title("Trend of Movie Duration Over the Years")
    plt.xlabel("Release Year")
    plt.ylabel("Duration (minutes)")
    plt.xticks(fontsize=10, rotation=45)
    plt.tight_layout()
    plt.show()

    # Hypothesis 2: More films have been produced in the USA over the years.
    usa_movies = movies[movies["country"].str.contains("United States")]

    plt.figure(figsize=(12, 6))
    sns.countplot(data=usa_movies, x="release_year",
                  order=usa_movies["release_year"].value_counts().index)
    plt.title("Number of Films Produced in the USA Over the Years")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Films")
    plt.xticks(fontsize=10, rotation=45)
    plt.tight_layout()
    plt.show()

    # Hypothesis 3: Longer titles often indicate a series.
    df["title_length"] = df["title"].apply(len)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="type", y="title_length")
    plt.title("Title Length by Type")
    plt.xlabel("Type")
    plt.ylabel("Title Length")
    plt.show()

def main():
    df = get_dataset("data/netflix_titles.csv")

    # analyze_hypotheses(df)
    predict_length(df, "Drama")

if __name__ == "__main__":
    main()
