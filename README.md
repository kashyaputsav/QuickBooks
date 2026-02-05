# Intelligent Expense Categorization System (QuickBooks‑Style)

**Project Duration:** Feb 2026
**Domain:** Machine Learning, NLP, FinTech Automation
**Tech Stack:** Python, Scikit‑learn, XGBoost, Pandas, NumPy, Flask, Jupyter Notebook

---

## 📌 Project Overview

Modern financial platforms like **QuickBooks** automatically categorize expenses (e.g., *Travel, Food, Utilities, Office Supplies*) from raw transaction data. This project replicates that real‑world system using **Natural Language Processing (NLP)** and **structured numerical features**, building a **production‑ready ML pipeline** for intelligent expense categorization.

The system takes raw transaction descriptions and metadata as input, predicts the most likely expense category, and assigns a **confidence score**. Low‑confidence predictions are flagged for **human review**, ensuring reliability and user trust — a core requirement in FinTech products.

---

## 🎯 Key Objectives

* Automate expense categorization at scale
* Combine text‑based NLP features with numerical transaction signals
* Handle uncertainty using confidence‑based decision logic
* Design the solution with **production deployment** in mind

---

## 🧠 Problem Statement

Manual expense categorization is:

* Time‑consuming
* Error‑prone
* Not scalable for large transaction volumes

The goal is to build a machine learning system that:

1. Accurately classifies expenses into predefined categories
2. Explains prediction confidence
3. Falls back to human intervention when the model is uncertain

---

## 📊 Dataset Description

The project uses a **synthetic but realistic QuickBooks-style expense dataset** designed to closely mirror real-world financial transaction data used in accounting platforms.

**Dataset File:** `quickbooks_synthetic_expenses_7k_diverse.csv`

### 🔢 Dataset Size

* ~7,000 transaction records
* Multi-class expense categorization problem

### 🧾 Columns Overview

**Text Feature**

* `description` – Raw transaction description containing merchant names, payment context, and notes (primary NLP signal)

**Numerical / Structured Features**

* `amount` – Transaction amount
* `payment_mode` – Mode of payment (card, UPI, cash, bank transfer, etc.)
* `merchant_type` – Merchant category signal
* `is_recurring` – Indicates recurring transactions
* `day_of_week` – Temporal pattern feature
* `month` – Seasonal spending signal

**Target Variable**

* `expense_category` – Ground truth expense class (e.g., Travel, Food, Utilities, Office Supplies, Entertainment, Healthcare, etc.)

### 🎯 Why This Dataset Works Well

* Mimics noisy, short-text financial descriptions
* Contains overlapping categories (realistic ambiguity)
* Supports confidence-based decision making
* Suitable for NLP + structured feature fusion

> The dataset is explored, cleaned, and modeled end-to-end inside the Jupyter Notebook.

---

## 🔧 Feature Engineering

### 1️⃣ Text Processing (NLP)

* Lowercasing & normalization
* Stopword removal
* TF‑IDF vectorization to convert transaction descriptions into numerical vectors

### 2️⃣ Numerical Feature Handling

* Scaling and normalization
* Handling missing values
* Feature alignment between training and inference

### 3️⃣ Feature Fusion

* Combined TF‑IDF vectors with numerical features into a single feature space

---

## 🤖 Model Architecture

* **Model Used:** XGBoost (Multi‑Class Classifier)
* **Why XGBoost?**

  * Handles mixed feature types well
  * Strong performance on structured + sparse data
  * Robust to noise and feature interactions

### Training Pipeline

1. Train‑validation split
2. Hyperparameter tuning
3. Multi‑class classification training
4. Performance evaluation

---

## 📈 Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision / Recall / F1‑Score (per class)
* Confusion Matrix

Model performance analysis is documented step‑by‑step in the Jupyter Notebook.

---

## ✅ Confidence Scoring & Human‑in‑the‑Loop

To make the system production‑ready:

* Model outputs class probabilities
* Maximum probability is treated as **confidence score**

### Decision Logic

* **High confidence →** Auto‑categorize
* **Low confidence →** Flag for manual review

This approach:

* Reduces incorrect auto‑classifications
* Improves long‑term trust
* Enables continuous feedback and retraining

---

## 🚀 Inference Service (Production‑Ready)

A Flask‑based inference service is implemented to:

* Accept transaction input via API
* Apply the same preprocessing pipeline
* Return:

  * Predicted category
  * Confidence score
  * Review flag (True / False)

This mirrors real‑world ML deployment patterns used in FinTech companies.

---

## 📁 Project Structure

```
project-root/
│
├── notebook/
│   └── Expense_Categorization.ipynb
│
├── app.py                 # Flask inference service
├── model.pkl              # Trained XGBoost model
├── vectorizer.pkl         # TF-IDF vectorizer
├── requirements.txt
├── README.md
└── data/
    └── transactions.csv
```

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Jupyter Notebook

```bash
jupyter notebook
```

Open `Expense_Categorization.ipynb` to explore training and evaluation.

### 3️⃣ Start Inference API

```bash
python app.py
```

---

## 🧪 Sample Prediction Output

```json
{
  "predicted_category": "Travel",
  "confidence": 0.87,
  "manual_review_required": false
}
```

---

## 💡 Real‑World Impact

This project closely aligns with:

* QuickBooks expense automation
* Invoice & transaction intelligence systems
* Enterprise‑grade ML pipelines

It demonstrates skills in:

* End‑to‑end ML system design
* NLP + structured data modeling
* Deployment‑ready ML thinking

---

## 🔮 Future Improvements

* Deep learning models (BERT‑based text encoders)
* Online learning with feedback loops
* Explainability using SHAP values
* Multi‑language transaction support

---

## 👤 Author

**Utsav Kashyap**
Data Scientist
Machine Learning | Backend | NLP

---

⭐ If you find this project useful, feel free to star the repository!
