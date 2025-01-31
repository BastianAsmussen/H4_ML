import sys

import pandas as pd
from pandas import DataFrame

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def sanitize_string(s: str) -> str:
    """
    Remove Danish stopwords from a given string.

    Args:
        s (str): The input string to sanitize.

    Returns:
        str: The sanitized string.
    """
    all_words = s.split()
    bad_words = stopwords.words("danish")

    # return [x for x in all_words if ]
    return ""

EXCEL_FILE = "data/Fagtabel_Excel_2024.xlsx"

def get_measurement_points(df: DataFrame, subject_number: int = None) -> list[str]:
    """
    Returns a list of measurement points for a given subject number.
    
    Args:
        df (DataFrame): The data frame to operate on.
        subject_number (int): The subject number to look up, if any.
        
    Returns:
        list[str]: A list of measurement points for the subject.
    """
    try:
        # Skip the header row and set proper column names.
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

        # Convert to numeric, coercing errors to NaN.
        df["NUMMER"] = pd.to_numeric(df["NUMMER"], errors="coerce")
        df["NIVEAU"] = pd.to_numeric(df["NIVEAU"], errors="coerce")

        df = df[~df["FAGKATEGORI"].isin(["Grundfag", "htx"])]
        df = df.sort_values("NIVEAU", ascending=False)

        # Filter for the specific result form.
        df = df[df["RESULTATFORM"] == "-/STA/7TRIN"]

        subject_row = df[df["NUMMER"]:]
        if subject_row.empty:
            return []

        # Get measurement points and split them into a list.
        measurement_points = str(subject_row["MÅLPINDE"].iloc[0])
        measurement_points = [point.strip() for point in measurement_points.split("\n") if point.strip()]

        return measurement_points

    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}", file=sys.stderr)

        return []

if __name__ == "__main__":
    SANITIZED = sanitize_string("Jeg kan godt lide ost og kage.")
    print(SANITIZED)

    df = pd.read_excel(EXCEL_FILE)

    ml = get_measurement_points(df)
    for point in ml:
        print(point)
