"""
models.py
---------
Defines two different LSTM-based sentiment classification models.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional


def build_simple_lstm(max_words, max_len):
    """Builds a simple single-layer LSTM model."""
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_bilstm_dropout(max_words, max_len):
    """Builds a more advanced BiLSTM with Dropout."""
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
