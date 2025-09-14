# Sentiment Analysis on IMDB Reviews using LSTM

This project performs **sentiment classification** on the IMDB movie reviews dataset using deep learning.  
It builds and compares two LSTM-based neural network models:  
- **Model 1:** A simple LSTM network  
- **Model 2:** A Bidirectional LSTM with Dropout and Dense layers

The workflow covers:
- Data exploration and preprocessing  
- Text tokenization and padding  
- Model training and validation  
- Evaluation using confusion matrices and classification reports  
- Performance visualization with accuracy plots  
- Model saving for future use
---

## 📊 Dataset Overview

- **Total samples:** 22,150 reviews  
- **Sentiment distribution:** ~50% Positive / ~50% Negative  
- **Average review length:** ~231 words  

---
## 🧠 Model Details

| Model | Architecture | Test Accuracy |
|------|--------------|-------------|
| **Model 1** | Simple LSTM (64 units) | **84.4%** |
| **Model 2** | Bidirectional LSTM (64 units) + Dropout | 81.8% |

---

## 📈 Performance Insights

- **Model 1 (Simple LSTM)** achieved slightly better generalization with **higher accuracy (84%)** and balanced precision/recall.  
- **Model 2 (BiLSTM + Dropout)** performed well but slightly underfit compared to Model 1 in this experiment, achieving ~82% accuracy.  
- Training curves show that **Model 1 converged faster and more stably**, whereas Model 2’s validation accuracy plateaued and fluctuated in later epochs.  

---

## 🧪 Evaluation Metrics

### Model 1 Confusion Matrix
|            | Predicted Negative | Predicted Positive |
|------------|------------------|------------------|
| **Actual Negative** | 1848 | 379 |
| **Actual Positive** | 311 | 1892 |

**F1-score:** 0.84

---

### Model 2 Confusion Matrix
|            | Predicted Negative | Predicted Positive |
|------------|------------------|------------------|
| **Actual Negative** | 1861 | 366 |
| **Actual Positive** | 439 | 1764 |

**F1-score:** 0.82

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/junaaaak/imdb_lstm.git
cd imdb_lstm
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Open the Jupyter Notebook:
```bash
jupyter notebook imdb_sentiment_analysis.ipynb
```

 You can then run the cells sequentially to reproduce the analysis and results.


