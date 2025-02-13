import re
from collections import Counter

import pandas as pd
from nltk.stem import PorterStemmer

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from imblearn.over_sampling import RandomOverSampler

import seaborn as sns
import matplotlib.pyplot as plt

DATASET_FILE = "data/spam-emails/spam.csv"
PORTER_STEMMER = PorterStemmer()

def read_dataset(path: str) -> pd.DataFrame:
    """
    Reads the dataset from a given CSV file path.

    Args:
        path (str): The URI to the CSV file.

    Returns:
        DataFrame: The data frame extracted from the path.
    """
    df = pd.read_csv(path)
    df.head()

    df = df.drop_duplicates(keep="first")

    return df

def sanitize_text(text: str) -> list[str]:
    """
    Sanitize a given text string.

    Args:
        text (str): The string to sanitize.

    Returns:
        list[str]: The text split into words.
    """
    text = text.lower()

    text = re.sub("\\W", " ", text)
    text = re.sub("\\s+(in|the|all|for|and|on)\\s+", " _connector_ ", text)

    words = re.split("\\s+", text)
    stemmed_words = [PORTER_STEMMER.stem(word=word) for word in words]

    return " ".join(stemmed_words)

def tokenize_text(text: str) -> list[str]:
    """
    Tokenize a given text string.

    Args:
        text (str): The text to tokenize.

    Returns:
        list[str]: The tokenized string.
    """
    text = re.sub("(\\W)", " \\1 ", text)

    return re.split("\\s+", text)

if __name__ == "__main__":
    df = read_dataset(DATASET_FILE)

    x = df["Message"].values
    y = df["Category"].values

    vectorizer = CountVectorizer(
        tokenizer=tokenize_text,
        ngram_range=(1, 2),
        min_df=0.006,
        preprocessor=sanitize_text
    )
    x = vectorizer.fit_transform(x)

    ros = RandomOverSampler(random_state=42)

    print('Original dataset shape:', Counter(y))

    # Fit predictor and target.
    x, y = ros.fit_resample(x, y)
    print('Modified dataset shape:', Counter(y))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    params = {"C": [0.2, 0.5], "kernel": ['linear', 'sigmoid']}
    cval = KFold(n_splits=2)
    model = SVC()

    TunedModel = GridSearchCV(model, params, cv=cval)
    TunedModel.fit(x_train, y_train)

    model.fit(x_train, y_train)

    accuracy = metrics.accuracy_score(y_test, TunedModel.predict(x_test))
    accuracy_percentage = 100 * accuracy

    sns.heatmap(confusion_matrix(y_test, TunedModel.predict(x_test)), annot=True, fmt="g")
    plt.xlabel("Predicted")
    plt.show()

    print(classification_report(y_test, TunedModel.predict(x_test)))

    mails = [
        "Hey, you have won a car !!!!. Conrgratzz",
        "Dear applicant, Your CV has been recieved. Best regards",
        "You have received $1000000 to your account",
        "Join with our whatsapp group",
        "Kindly check the previous email. Kind Regard"
    ]

    print(f"Checking {len(mails)} mails for spam or ham...")
    for mail in mails:
        is_spam = TunedModel.predict(vectorizer.transform([mail]).toarray())[0]

        print(f"- \"{mail}\" looks like {is_spam}.")
