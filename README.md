# Anomaly Detection on the NSL-KDD Dataset

A hands-on exploration of classical and deep learning techniques for identifying network intrusions.

---

## Overview

This project applies and compares multiple anomaly detection techniques on the NSL-KDD dataset — a benchmark dataset for network intrusion detection. It includes exploratory data analysis, feature engineering, classical machine learning models, and deep learning architectures. The project also serves as a reusable template for structuring future anomaly detection work.

Along the way, I’m building personal fluency with both the conceptual underpinnings and practical implementations of anomaly detection — making this a dual-purpose learning and portfolio project.

---

## Dataset

- **Name**: NSL-KDD (derived from KDD’99)
- **Source**: [UNB CIC Repository](https://www.unb.ca/cic/datasets/nsl.html)
- **Size**: ~125,000 rows, 41 features
- **Task**: Binary anomaly detection (normal vs. attack) with multiclass subtype labels
- **Why it’s interesting**: NSL-KDD is widely used for network intrusion detection benchmarking and includes a range of attack types under realistic conditions of class imbalance.

---

## Methods Explored

| Method              | Category         | Tools Used            |
|---------------------|------------------|------------------------|
| Logistic Regression | Baseline         | `scikit-learn`         |
| Isolation Forest    | Unsupervised     | `scikit-learn`, `pyod` |
| One-Class SVM       | Unsupervised     | `scikit-learn`         |
| Autoencoder         | Deep Learning    | `tensorflow`, `keras`  |
| TBD...              | Future methods   | ...                    |

---

## Learning Objectives

- Understand key categories of anomaly detection
- Evaluate performance of classical vs. deep learning models
- Develop a modular, reusable structure for anomaly detection projects
- Communicate technical findings clearly with visuals and metrics

---

## Results Snapshot

_(WIP: to be updated as experiments are run)_

| Model             | ROC-AUC | Precision | Recall | F1   |
|------------------|---------|-----------|--------|------|
| Logistic Reg.     | TBD     | TBD       | TBD    | TBD  |
| Isolation Forest  | TBD     | TBD       | TBD    | TBD  |
| Autoencoder       | TBD     | TBD       | TBD    | TBD  |

---

## Repo Structure
anomaly-nsl-kdd/
│

├── README.md

├── requirements.txt

├── .gitignore

│

├── data/

│   ├── README.md

│   ├── raw/

│   └── processed/

│

├── notebooks/

│   ├── 01_exploration.ipynb

│   ├── 02_feature_engineering.ipynb

│   ├── 03_modeling.ipynb

│   └── 04_evaluation.ipynb

│

├── src/

│   ├── __init__.py

│   ├── data_prep.py

│   ├── modeling.py

│   └── utils.py

│

├── reports/

│   ├── figures/

│   └── summary.md

│

└── app/

    └── streamlit_app.py  # Optional for dashboard deployment
    


---

## How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/anomaly-nsl-kdd.git
cd anomaly-nsl-kdd

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook

