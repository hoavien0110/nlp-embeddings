import re
import pandas as pd

def load_and_clean_data(file_path):
    """Load and clean the data."""
    df = pd.read_excel(file_path)
    df.rename(columns={"Character": "Label", "Text": "sentence"}, inplace=True)
    df = df[df['sentence'].apply(lambda x: isinstance(x, str))]  # Remove non-string sentences

    def clean_text(text):
        text = re.sub(r'\s+', '', text)  # Remove extra spaces
        text = re.sub(r'。', '', text)  # Remove specific punctuation
        return text

    df['sentence_cleaned'] = df['sentence'].apply(lambda x: clean_text(x))
    df['BinaryLabel'] = df['Label'].apply(lambda x: 1 if x == 'Nom' else 0)
    return df
