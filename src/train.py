"""
train.py
--------
Trains both models, evaluates them, and compares performance.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from preprocess import load_and_preprocess_data
from models import build_simple_lstm, build_bilstm_dropout

MAX_WORDS = 10000
MAX_LEN = 200

# Load data
X_train, X_test, y_train, y_test, tokenizer, label_encoder = load_and_preprocess_data(
    "data/imdb_reviews.csv", MAX_WORDS, MAX_LEN
)

# Build models
model1 = build_simple_lstm(MAX_WORDS, MAX_LEN)
model2 = build_bilstm_dropout(MAX_WORDS, MAX_LEN)

# Train models
history1 = model1.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
history2 = model2.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Save models
model1.save("../saved_models/model1_simple_lstm.h5")
model2.save("../saved_models/model2_bilstm_dropout.h5")

# Predictions
y_pred1 = (model1.predict(X_test) > 0.5).astype("int32")
y_pred2 = (model2.predict(X_test) > 0.5).astype("int32")

# Evaluation function
def evaluate_model(y_true, y_pred, title):
    print(f"\n🔹 {title} Evaluation Report")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

    # Plot confusion matrix as heatmap
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'])
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Evaluate both models
evaluate_model(y_test, y_pred1, "Model 1 (Simple LSTM)")
evaluate_model(y_test, y_pred2, "Model 2 (BiLSTM + Dropout)")

# Accuracy comparison plot
def plot_history(history, label):
    plt.plot(history.history['accuracy'], label=f'{label} Train Acc')
    plt.plot(history.history['val_accuracy'], label=f'{label} Val Acc')

plt.figure(figsize=(10, 5))
plot_history(history1, "Model 1")
plot_history(history2, "Model 2")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
