import sys
import re
from collections import Counter

import pandas as pd
from pandas import DataFrame

import nltk
from nltk.corpus import stopwords

from wordcloud import WordCloud
import matplotlib.pyplot as plt

EXCEL_FILE = "data/Fagtabel_Excel_2024.xlsx"
STOPWORDS = set(stopwords.words("danish"))

def sanitize_string(text: str) -> str:
    """
    Remove Danish stopwords from a given string.

    Args:
        text (str): The input string to sanitize.

    Returns:
        str: The sanitized string.
    """
    words = re.findall(r"[a-zA-ZæøåÆØÅ]+(?:'[a-zA-ZæøåÆØÅ]+)?", text.lower())

    return " ".join(w for w in words if w not in STOPWORDS)

def get_measurement_points(input_df: DataFrame, subject_number: int = None) -> list[str]:
    """
    Returns a list of measurement points for a given subject number.
    
    Args:
        input_df (DataFrame): The data frame to operate on.
        subject_number (int, optional): The subject number to look up. Defaults to None.
        
    Returns:
        list[str]: A list of measurement points for the subject.
    """
    # Create a copy to avoid modifying the original data frame.
    processed_df = input_df.copy()

    # Set headers.
    processed_df.columns = processed_df.iloc[0]
    processed_df = processed_df.iloc[1:].reset_index(drop=True)

    processed_df["NUMMER"] = pd.to_numeric(processed_df["NUMMER"], errors="coerce")
    processed_df["NIVEAU"] = pd.to_numeric(processed_df["NIVEAU"], errors="coerce")

    processed_df = processed_df[~processed_df["FAGKATEGORI"].isin(["Grundfag", "htx"])]
    processed_df = processed_df[processed_df["RESULTATFORM"] == "-/STA/7TRIN"]
    processed_df = processed_df.sort_values("NIVEAU", ascending=False)

    subject_row = processed_df
    if subject_number:
        subject_row = processed_df[processed_df["NUMMER"] == subject_number]

    if subject_row.empty:
        return []

    measurement_points_text = str(subject_row["MÅLPINDE"].iloc[0])

    return [point.strip() for point in measurement_points_text.split("\n") if point.strip()]

def generate_wordcloud(words: list[str]) -> None:
    """
    Generate and display a word cloud visualization from a list of words.
    
    Args:
        words (list[str]): List of words to include in the word cloud.
    """
    text = " ".join(words)
    wc = WordCloud(background_color='black', width=800, height=500).generate(text)
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(wc)
    plt.title("Word Cloud of Measurement Points")
    plt.show()

def plot_word_frequencies(words: list[str], top_n: int = 30) -> None:
    """
    Create a horizontal bar plot of word frequencies.
    
    Args:
        words (list[str]): List of words to analyze.
        top_n (int, optional): Number of top words to display. Defaults to 30.
    """
    word_counts = Counter(words)
    common_words = word_counts.most_common(top_n)

    words, counts = zip(*common_words)
    plt.figure(figsize=(12, 8))
    plt.barh(words, counts, color="blue")
    plt.gca().invert_yaxis()
    plt.xlabel("Occurrences")
    plt.ylabel("Words")
    plt.title(f"Top {top_n} Most Common Words in Measurement Points")
    plt.tight_layout()
    plt.show()

def main() -> None:
    """Execute the main program flow."""
    try:
        nltk.download("stopwords", quiet=True)
        df = pd.read_excel(EXCEL_FILE)

        # Process words.
        sanitized_words = [
            word
            for point in get_measurement_points(df)
            for word in sanitize_string(point).split()
        ]

        if not sanitized_words:
            print("Warning: No words found after processing!", file=sys.stderr)

            return

        # Generate visualizations.
        generate_wordcloud(sanitized_words)
        plot_word_frequencies(sanitized_words)

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

        sys.exit(1)

if __name__ == "__main__":
    main()
