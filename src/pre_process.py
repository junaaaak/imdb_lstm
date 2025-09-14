"""
preprocess.py
-------------
Handles dataset loading, text preprocessing, tokenization, and splitting.
"""

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(csv_path, max_words=10000, max_len=200):
    """Loads dataset, tokenizes reviews, pads sequences, encodes labels.

    Args:
        csv_path (str): Path to the CSV file.
        max_words (int): Max number of words to keep in tokenizer vocabulary.
        max_len (int): Max sequence length for padding.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, tokenizer, label_encoder)
    """
    df = pd.read_csv(csv_path)
    df.dropna(subset=['review', 'sentiment'], inplace=True)

    # Encode labels
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

    # Tokenize and pad
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['review'])
    X = tokenizer.texts_to_sequences(df['review'])
    X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')

    y = df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, tokenizer, label_encoder
