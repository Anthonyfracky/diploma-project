# Bachelor Thesis Project: Detection of Malicious Network Traffic Using Machine Learning

## Project Overview

This repository contains the source code for a bachelor’s thesis focused on detecting malicious network traffic. The primary objective is to develop and evaluate machine learning models capable of performing both binary classification (Benign vs Malicious) and multiclass classification (by type of attack) using the real-world Aposemat IoT-23 dataset. The project includes all necessary stages: from raw data aggregation and preprocessing to model training, evaluation, and result visualization.

## Workflow Summary

### 1. **Data Aggregation**
The `data_agregator.py` script collects and merges all raw CSV files located in the `data/` directory into a unified dataset (`dataset.csv`).

### 2. **Data Preprocessing**
- `data_preprocessing.py` handles the binary classification task. It removes columns with a high percentage of missing data, consolidates all malicious subtypes into a single “Malicious” label, filters invalid records, and saves the result as `data_upd.csv`.
- `data_preprocessing_multiclass.py` is designed for multiclass classification. It encodes specific attack types into numeric labels, eliminates unusable or irrelevant fields, handles missing data, and outputs the cleaned dataset as `dataset_upd_multiclass.csv`.

### 3. **Exploratory Data Analysis**
The script `data_analisys.py` conducts preliminary dataset analysis, including:
- General structure (dimensions, dtypes),
- Missing values overview,
- Class distribution visualization,
- Descriptive statistics for numerical features.
The findings are exported as a text report and image files.

### 4. **Model Training**
- `model_train.py` supports binary classification and offers both classic train-test evaluation and cross-validation. Implemented models include: Logistic Regression, Naive Bayes, Random Forest, AdaBoost, XGBoost, and LightGBM. It produces confusion matrices, feature importance plots, and metric summaries.
- `model_train_multiclass.py` is used for multiclass classification. Each model is trained and tested against labeled attack types. The script generates confusion matrices, full classification reports, training time, and comparative evaluation of models such as LightGBM, XGBoost, AdaBoost, Random Forest, Naive Bayes, and Logistic Regression.

## Repository Structure

```

├── data_agregator.py            # Merges all CSVs
├── data_preprocessing.py        # Binary preprocessing pipeline
├── data_preprocessing_multiclass.py  # Multiclass preprocessing pipeline
├── data_analisys.py             # Dataset diagnostics and EDA
├── model_train.py               # Binary classification models
├── model_train_multiclass.py    # Multiclass classification models
└── requirements.txt             # Python dependencies

```

## Instructions for Running

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Combine raw CSV files into one dataset

```bash
python data_agregator.py
```

### Step 3: Run preprocessing

- **Binary classification:**
  ```bash
  python data_preprocessing.py
  ```

- **Multiclass classification:**
  ```bash
  python data_preprocessing_multiclass.py
  ```

### Step 4: Perform optional data analysis

```bash
python data_analisys.py
```

### Step 5: Train models

- **Binary classification:**
  ```bash
  python model_train.py
  ```

- **Multiclass classification:**
  ```bash
  python model_train_multiclass.py
  ```

## Outputs

- Binary and multiclass confusion matrices, visualizations of feature importance and class distribution are saved under `confusions/` and `confusions_multiclass/`.
- A textual summary of model performance is available as `all_models_report.txt`.

---

**Author:** Anton Fadieiev  
**Institution:** Lviv Polytechnic National University  
**Year:** 2025
