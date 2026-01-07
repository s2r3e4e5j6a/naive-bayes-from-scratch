# Naive Bayes Classifier – From Scratch (Teaching Repository)

This repository demonstrates **how a Naive Bayes classification model is developed from scratch**
using **Python, NumPy, and Pandas**, without using any machine learning libraries like scikit-learn.

This project is designed **for learning and teaching purposes**, especially for beginners who want
to understand what happens *inside* a machine learning model.

---

## 📌 Why Naive Bayes?

Naive Bayes is one of the simplest and most intuitive machine learning algorithms.
It is based on **probability theory (Bayes’ Theorem)** and works surprisingly well
for text classification problems like spam detection.

---

## ❌ Why NOT scikit-learn?

Libraries like scikit-learn hide the internal logic.

In this project:
- We calculate probabilities manually
- We build the model step by step
- We understand how **training** and **prediction** actually work

---

## 📂 Project Structure
```
naive-bayes-from-scratch/
│
├── data/
│   └── corpus_small.csv        # Small text dataset (spam / ham)
│
├── src/
│   └── naive_bayes.py          # Naive Bayes implementation from scratch
│
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

