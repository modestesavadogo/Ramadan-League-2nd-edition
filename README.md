# 🌙 Ramadan League — 2nd Edition · Team Submission

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Submitted-success?style=for-the-badge)

**A Data Science competition organized as part of an academic learning journey by CIT/INPT.**

</div>

---

## 👥 Team

| Name | GitHub |
|------|--------|
| Modeste Savadogo | [@modestesavadogo](https://github.com/modestesavadogo) |
| Diallo Mamadou | *(add GitHub handle)* |

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problems & Solutions](#-problems--solutions)
  - [1. The Hidden Leak](#1-the-hidden-leak--20)
  - [2. Sentiment Sleuth](#2-sentiment-sleuth--30)
  - [3. Buggy Logistic Regression](#3-buggy-logistic-regression--20)
  - [4. Taxi Time Challenge](#4-taxi-time-challenge--30)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)

---

## 🧭 Overview

This repository contains our team's solutions to the **Ramadan League Data Science Competition (2nd Edition)** — a 4-problem challenge covering time-series leakage detection, NLP classification, ML debugging, and geospatial regression.

The competition emphasizes deep understanding over quick results. Each solution was built through personal research, experimentation, and documented reasoning.

---

## 🧩 Problems & Solutions

### 1. The Hidden Leak · 20%

> **Task:** Identify and eliminate data leakage in a time-series forecasting pipeline, then design a leak-free evaluation protocol.

**Our Approach:**
- Carefully analyzed the pipeline for temporal leakage points (e.g., future data used during scaling/feature engineering)
- Identified specific leakage sources and explained their impact on model evaluation
- Proposed a rigorous walk-forward validation protocol to prevent leakage

**Deliverables:** `report.md` *(+ optional `analysis.ipynb`)*

---

### 2. Sentiment Sleuth · 30%

> **Task:** Build a multiclass tweet sentiment classifier (negative / neutral / positive) from labeled text data.

**Our Approach:**
- Preprocessed raw tweet text (tokenization, stopword removal, normalization)
- Explored multiple classification strategies (TF-IDF + classical ML, fine-tuned transformers)
- Evaluated with macro F1-score across all three classes

**Deliverables:** `solution.ipynb` *(+ optional `train.py`)*

---

### 3. Buggy Logistic Regression · 20%

> **Task:** Debug and correct a broken from-scratch logistic regression implementation, then verify expected model behavior.

**Deliverables:** `problem.py` (fixed) + `notes.md`

---

### 4. Taxi Time Challenge · 30%

> **Task:** Predict NYC taxi trip duration using temporal and geospatial features with strong feature engineering.

**Our Approach:**
- Extracted temporal features (hour, day of week, rush hour flags)
- Engineered geospatial features (Haversine distance, bearing, pickup/dropoff clusters)
- Trained and compared regression models with cross-validation

**Deliverables:** `solution.ipynb` *(+ optional `train.py`, `inference.py`)*

---

## 📁 Repository Structure

```
Ramadan-League-2nd-edition/
│
├── 1.The Hidden Leak/
│   ├── report.md              # Leakage analysis & fix protocol
│   └── analysis.ipynb         # (optional) Pseudo-code & sketches
│
├── 2. Sentiment Sleuth/
│   ├── solution.ipynb         # Full NLP pipeline & results
│   └── train.py               # (optional) Reproducible training script
│
├── 3. Buggy Logistic Regression/
│   ├── problem.py             # Fixed implementation
│   └── notes.md               # Bug descriptions & explanations
│
├── 4. Taxi Time Challenge/
│   ├── solution.ipynb         # Feature engineering & modeling
│   ├── train.py               # (optional) Training pipeline
│   └── inference.py           # (optional) Inference script
│
└── README.md                  # This file
```

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **Notebooks:** Jupyter
- **ML / DL:** scikit-learn, XGBoost, (optionally) HuggingFace Transformers
- **NLP:** NLTK / spaCy / TF-IDF
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/modestesavadogo/Ramadan-League-2nd-edition.git
cd Ramadan-League-2nd-edition

# 2. Install dependencies
pip install -r requirements.txt   # (if provided)

# 3. Open any notebook
jupyter notebook
```

---

<div align="center">

*Built with curiosity and care during Ramadan 2026 🌙*

</div>
