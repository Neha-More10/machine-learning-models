# Bank Customer Churn Prediction

A machine learning project that predicts whether a banking customer is likely to leave (churn) based on demographic, account, and engagement data.

This project demonstrates a complete classification workflow using Python and scikit-learn, including:

- Synthetic data generation
- Data preprocessing
- Feature engineering
- Logistic Regression modelling
- Model evaluation
- Model interpretability

---

## Business Problem

Customer churn directly impacts revenue and customer retention costs.

The goal of this project is to identify customers who are likely to leave a bank based on factors such as:

- Credit score
- Age
- Account balance
- Product ownership
- Customer activity
- Salary
- Geographic location

The model predicts:

- `0` = Customer stayed
- `1` = Customer churned

---

## Solution Overview

The project follows a standard machine learning workflow:

```text
Raw Customer Data
        │
        ▼
Preprocessing
(Scaling + Encoding)
        │
        ▼
Logistic Regression
        │
        ▼
Churn Probability
        │
        ▼
Customer Retention Insights
```

---

## Dataset

The repository includes a synthetic banking dataset generated using NumPy.

Features include:

| Feature | Description |
|----------|-------------|
| credit_score | Customer credit score |
| country | Customer country |
| gender | Customer gender |
| age | Customer age |
| tenure | Years with the bank |
| balance | Account balance |
| products_number | Number of products held |
| credit_card | Credit card ownership |
| active_member | Customer activity status |
| estimated_salary | Estimated annual salary |

Target:

| Column | Description |
|---------|-------------|
| churn | 1 = Churned, 0 = Stayed |

The synthetic dataset is designed to resemble common banking churn datasets while remaining lightweight and easy to run locally.

---

## Machine Learning Pipeline

The model uses a Scikit-Learn Pipeline to ensure preprocessing and training are applied consistently.

### Numeric Features

Numeric features are standardised using:

```python
StandardScaler()
```

### Categorical Features

Categorical features are encoded using:

```python
OneHotEncoder()
```

### Model

```python
LogisticRegression()
```

Pipeline structure:

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])
```

---

## Model Evaluation

The training script reports:

- Accuracy
- ROC AUC
- Precision
- Recall
- F1 Score
- Confusion Matrix

It also displays the most influential features driving churn predictions.

Example:

```text
Rows: 1,200
Churn rate: 17.1%

Accuracy: 0.677
ROC AUC: 0.731
```

---

## Project Structure

```text
.
├── data/
│   └── bank_churn_sample.csv
│
├── src/
│   ├── make_sample_data.py
│   └── train_logistic_regression.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Getting Started

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate sample data

```bash
python src/make_sample_data.py
```

### Train the model

```bash
python src/train_logistic_regression.py
```

---

## Skills Demonstrated

- Python
- pandas
- NumPy
- scikit-learn
- Data preprocessing
- Feature engineering
- Logistic Regression
- Machine Learning Pipelines
- Classification modelling
- Model evaluation
- Explainable AI

---

## Future Improvements

Potential enhancements:

- Random Forest Classifier
- XGBoost
- Hyperparameter tuning
- Cross-validation
- Feature importance visualisation
- Model deployment with FastAPI
- Streamlit dashboard

---

## Disclaimer

This project uses synthetic data for educational and portfolio purposes. The workflow mirrors real-world churn prediction pipelines used in banking, fintech, and subscription-based businesses.
