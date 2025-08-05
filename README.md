# HR Survey Sentiment Analysis

### Automating Employee Sentiment Detection to Reduce Turnover

## Project Overview

This project aims to proactively reduce employee turnover by **automating sentiment analysis of HR survey responses**. The system processes textual feedback from employees and classifies their sentiment (positive, neutral, or negative), flagging high-risk cases for follow-up. This allows HR departments to respond in **real-time**, hold mitigation meetings, and improve employee satisfaction before dissatisfaction leads to resignation.

A **pre-trained RoBERTa model**, originally trained on 124M+ tweets, was further **fine-tuned for this domain-specific use case**, enhancing prediction accuracy for professional feedback and HR language.

---

## Dataset

The project uses an HR survey dataset with feedback columns such as:

- `summary`
- `pros`
- `cons`
- `advice-to-mgmt`

After cleaning and preprocessing:

- Irrelevant metadata is removed.
- Feedback fields are merged into a unified `full_text` column.
- POS tagging is applied for richer input features.
- A numeric `sentiment_score` is used and mapped to class labels:
  - `0` = Negative
  - `1` = Neutral
  - `2` = Positive

---

## Models Used

Two transformer models are evaluated:

1. `roberta-base` – A general-purpose language model.
2. `cardiffnlp/twitter-roberta-base-sentiment` – Pretrained on 124M+ tweets, highly effective for sentiment detection.

Both models are:
- Evaluated in pretrained state.
- Fine-tuned on the HR dataset.

---

## Model Training & Evaluation

Models are trained using Hugging Face's `Trainer` API with:

- 3 epochs
- Learning rate: `2e-5`
- Batch size: `8`
- Mixed precision (`fp16`) for performance

### Evaluation Metrics:

- **Accuracy**
- **F1 Score (Weighted)**
- **Classification Report (Precision, Recall, F1 per class)**

Error analysis CSVs are generated to diagnose misclassifications.

---

## Results Summary

The fine-tuned `twitter-roberta-base-sentiment` model performed best due to its training on informal and emotionally expressive text (tweets), aligning well with employee feedback data.

- Improved accuracy after fine-tuning
- Clear identification of high-risk negative responses
- Practical use in real-time HR monitoring systems
