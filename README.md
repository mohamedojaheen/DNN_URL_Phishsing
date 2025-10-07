# DNN Detecting Phishing URLs

This project focuses on **detecting phishing websites** using **machine learning**.  
It continues from earlier phases of data gathering and preprocessing to model training, evaluation, and analysis.

The dataset used comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites), which contains features extracted from legitimate and phishing websites.  
The dataset was collected by **PhishTank Archive**, **MillerSmiles Archive**, and **Google’s search operators**.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Author](#author)
- [License](#license)

---

## Overview

Phishing is one of the most common cybersecurity threats that trick users into revealing sensitive information such as passwords, credit card details, or login credentials.  
This project aims to **detect phishing URLs automatically** using **data-driven analysis and machine learning** techniques.

The notebook (`Detecting_Phishing_URL_Phase2&3.ipynb`) demonstrates:
- Downloading and preparing the dataset.
- Exploratory data analysis (EDA).
- Feature selection and preprocessing.
- Model training and testing.
- Evaluation using accuracy, precision, recall, and F1-score.

---

## Dataset

- **Source:** [Phishing Websites Data Set - UCI Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites)  
- **Format:** `CSV`  
- **Total Samples:** ~11,000  
- **Features:** 30+ attributes derived from website characteristics such as:
  - URL length
  - Presence of “@” symbol
  - Number of dots (`.`)
  - SSL certificate validity
  - Domain registration length
- **Target Variable:**  
  - `1` → Legitimate  
  - `-1` → Phishing

---

## Project Workflow

1. **Data Acquisition**
   - Downloads dataset directly from UCI using Python `requests`.
   - Saves the file locally as `data.csv`.

2. **Data Loading**
   - Reads dataset into a Pandas DataFrame.
   - Checks for missing values and verifies data types.

3. **Preprocessing**
   - Cleans the dataset (handles missing or invalid entries).
   - Encodes categorical values into numerical form.
   - Splits data into training and testing sets.

4. **Model Building**
   - Trains several ML models (e.g., Decision Tree, Random Forest, Logistic Regression, SVM).
   - Tunes hyperparameters and evaluates performance.

5. **Model Evaluation**
   - Compares models using:
     - Accuracy
     - Precision
     - Recall
     - F1-score
   - Visualizes confusion matrices and performance graphs.

6. **Result Analysis**
   - Identifies the best-performing model.
   - Interprets important features influencing phishing detection.

---

## Machine Learning Models

| Model | Description | Key Notes |
|--------|--------------|-----------|
| Decision Tree | Simple interpretable model for classification | Often used as baseline |
| Random Forest | Ensemble of decision trees | High accuracy and robustness |
| Logistic Regression | Linear model for binary classification | Fast and simple |
| SVM | Separates phishing and legitimate URLs using optimal hyperplane | Effective for complex patterns |

---

## 📈 Results (Example Summary)

| Metric | Best Model (Random Forest) |
|--------|-----------------------------|
| Accuracy | 97.5% |
| Precision | 96.8% |
| Recall | 97.1% |
| F1-Score | 97.0% |

> *(Values are approximate and depend on training split and tuning parameters.)*

---

## Installation

Clone this repository:

```bash
git clone https://github.com/<your-username>/Detecting-Phishing-URL-Phase2-3.git
cd Detecting-Phishing-URL-Phase2-3
````

---

## Usage

1. Open the notebook:

   ```bash
   jupyter notebook Detecting_Phishing_URL_Phase2&3.ipynb
   ```

2. Run all cells sequentially to:

   * Download the dataset
   * Preprocess the data
   * Train ML models
   * Evaluate their performance

3. (Optional) Modify model parameters in the notebook to experiment with performance.

---

## Dependencies

Install all required libraries using:

```bash
pip install -r requirements.txt
```

### Main Packages Used:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `requests`
