# data_preprocessing.py

import pandas as pd
import spacy
from pathlib import Path

def load_and_preprocess_data():
    # Load the CSV file (adjust the path if needed)
    file_path = Path(__file__).parent / "impo.csv"
    df = pd.read_csv(file_path, skiprows=4)

    # Rename columns explicitly
    df.columns = ["text", "text_en", "cmp_code", "eu_code"]

    # Filter valid triplets and labels
    df = df.dropna(subset=['cmp_code'])
    df = df[df['cmp_code'].apply(lambda x: x.isdigit() and len(x) == 3)]

    # Filter rows where cmp_code starts with '4' (Economy domain)
    df_economy = df[df['cmp_code'].str.startswith('4')]

    # Load SpaCy model
    nlp = spacy.load("en_core_web_sm")

    # Function to preprocess the text (removes stop words and punctuation)
    def preprocess_text(text):
        doc = nlp(text)
        return ' '.join([token.text for token in doc if not token.is_stop and not token.is_punct])

    # Apply the preprocessing function to the 'text' column
    df_economy['processed_text'] = df_economy['text'].apply(preprocess_text)

    # Function to extract relationships (triplets) using dependency parsing
    def extract_relationships(text):
        doc = nlp(text)
        relationships = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ("nsubj", "dobj"):  # Subject and object detection
                    subject = token.text
                    verb = token.head.text
                    obj = [child for child in token.head.children if child.dep_ == "dobj"]
                    obj = obj[0].text if obj else ""
                    if subject and verb and obj:  # Ensure valid triplet
                        relationships.append((subject, verb, obj))
        return relationships

    # Apply the triplet extraction function
    df_economy['relationships'] = df_economy['text'].apply(extract_relationships)

    # Filter out any rows with empty relationships
    df_economy = df_economy[df_economy['relationships'].apply(lambda x: len(x) > 0)]

    # Print a sample of the extracted economy domain triplets
    print(df_economy[['text', 'relationships']].head())

    return df_economy
