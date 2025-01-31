# Udfordringer:
# - Regnarket indholder 4 faneblade: ét til fælles fag, ét til programmering, ét
#   til infrastruktur og ét til IT-supporter.
# - Arket indeholder (af én eller anden grund) fag med kategorier som Grundfag
#   og htx. Disse vil vi gerne filtrere væk.
# - Nogle fag findes flere gange. Bla. én gang for hvert niveau (2-4). Vi ønsker
#   blot at kigge på højeste niveau, som vi finde for hvert fag.
# - Nogle fag findes endda flere gang med samme niveau, men med forskellige
#   Resultatformer. Vi ønsker at finde den med resultatformen "-/STA/7TRIN".
# - Målepindene står i ét felt, formateret med linie skift.

import pandas as pd
import re

EXCEL_FILE = "data/Fagtabel_Excel_2024.xlsx"
df = pd.read_excel(EXCEL_FILE, sheet_name="Datatekniker med speciale i pro")

def get_stopwords() -> set:
    with open("data/stopwords.txt", "r") as file:
        stopwords = { line.strip().lower() for line in file.readlines() }

    return stopwords

stopwords = get_stopwords()

def get_milestone(fagnr: int) -> list[str]:
    fagnr = str(fagnr)

    for _, row in df.iterrows():
        if str(row.iloc[0]).strip() == fagnr:
            text = row.iloc[2]
            if isinstance(text, str):
                result = re.split(r'(?=\dd+\.\s)', text)
                return result

    print(f"No row found for NUMMER {fagnr}")

    return []

def get_milestone_no_nummer(fagnr: int) -> list[str]:
    fagnr = str(fagnr)

    for _, row in df.iterrows():
        if str(row.iloc[0]).strip() == fagnr:
            text = row.iloc[2]
            if isinstance(text, str):
                result = re.split(r'(?=\dd+\.\s)', text)
                result = [re.sub(r'^\dd+\.\s', '', part) for part in result]
                return result

    print(f"No row found for NUMMER {fagnr}")
    return []

def get_milestone_stopwords(fagnr: int) -> list[str]:
    fagnr = str(fagnr)

    for _, row in df.iterrows():
        if str(row.iloc[0]).strip() == fagnr:
            text = row.iloc[2]
            if isinstance(text, str):
                result = re.split(r'(?=\dd+\.\s)', text)

                filtered_result = [
                    ' '.join([word for word in part.split() if word.strip().lower() not in stopwords]).strip()
                    for part in result
                ]

                return [part for part in filtered_result if part]

    print(f"No row found for NUMMER {fagnr}")

    return []


def select_milestone() -> None:
    print("List of all Nummer and Titles:")
    for _, row in df.iterrows():
        nummer = row.iloc[0]
        title = row.iloc[1]
        print(f"{nummer} | {title}")

    selected_fagnr = int(input("\nNummer: "))

    result = get_milestone(selected_fagnr)

    if result:
        print(f"\nResults for fagnr {selected_fagnr}:")
        for m in result:
            print(m)


def all_milestones_nummer() -> None:
    for _, row in df.iterrows():
        nummer = str(row.iloc[0]).strip()

        if nummer.isnumeric():
            print(f"\nResults for fagnr {nummer}:")
            result = get_milestone(nummer)

            if result:
                for m in result:
                    print(m)
            else:
                print(f"No maalepinde found for {nummer}.")
        else:
            continue

#select_maalepinde()

ml = get_milestone_stopwords(1595)
#sw_test = get_milestone(16484)

print("Results for fagnr 1595:")
for m in ml:
    print(m)

#print("\nResults for fagnr 16484:")
#for m in sw_test:
#    print(m)
