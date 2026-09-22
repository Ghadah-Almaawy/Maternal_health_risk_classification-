# Maternal Health Risk Classification 

A data mining project that classifies pregnant women into three maternal health risk levels (**Low / Mid / High**) using physiological indicators, comparing six different machine learning classifiers to find the most reliable model.

> Course project — DS312: Data Mining.
> This was a **group project**; this repository reflects my contribution to the shared codebase.

**Live Demo:** https://GhadahAlmaawy.github.io/maternal-health-risk-classification/

##  Problem Statement

Maternal mortality remains a critical global health issue, and early risk detection can help prioritize care for high-risk pregnancies. This project uses IoT-collected physiological data (age, blood pressure, blood sugar, body temperature, heart rate) from hospitals and community clinics to predict a pregnant woman's health risk level.

**Dataset:** [Maternal Health Risk Data Set](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data/data) (Kaggle) — 1,014 records.

## Objective

Build and compare multiple classification models to predict maternal health risk level (Low/Mid/High), and identify the most accurate and reliable model for this task.

##  Project Workflow

1. **EDA** — inspected structure, data types, and summary statistics; visualized variable trends by risk level (e.g. age, blood pressure, blood sugar).
2. **Data Preprocessing** — checked for missing values and duplicates, fixed data types, and handled outliers per-feature using the IQR method with domain-specific thresholds.
3. **Class Imbalance Handling** — applied **SMOTE** (via `imbalanced-learn`) to balance the Low/Mid/High risk classes before training.
4. **Feature Engineering** — Min-Max scaling for numerical features and One-Hot Encoding for categorical features via a reusable preprocessing pipeline.
5. **Model Building** — trained and evaluated **six classifiers**:
   - K-Nearest Neighbors (KNN)
   - Naïve Bayes
   - Random Forest
   - Decision Tree
   - Support Vector Machine (SVM)
   - Artificial Neural Network (ANN, Keras/TensorFlow)
6. **Evaluation** — confusion matrices, classification reports, ROC curves, and a final comparison using **Jaccard index, F1-score, and Log Loss**.

##  Results

| Model | Accuracy | F1-score | Jaccard | Log Loss |
|---|---|---|---|---|
| KNN | 67.9% | 0.677 | 0.517 | 3.093 |
| Naïve Bayes | 60.4% | 0.601 | 0.442 | **0.774** |
| Decision Tree | 68.7% | 0.667 | 0.516 | 1.198 |
| **Random Forest** | **77.6%** | **0.774** | **0.643** | 0.816 |
| SVM | 47.0% | 0.460 | 0.300 | 1.005 |
| ANN | 55.2% | 0.526 | 0.369 | 0.965 |

**Random Forest** was the best-performing model on both F1-score and Jaccard index, correctly classifying 77.6% of cases, with especially strong precision on the "Low Risk" class (93%). Naïve Bayes achieved the lowest Log Loss, meaning its predicted probabilities were the best calibrated, despite lower raw accuracy.

## Interactive Demo

A browser-based version lets you adjust six clinical measurements (age, blood pressure, blood sugar, body temperature, heart rate) with sliders and get an instant risk-level prediction, with confidence shown across all three risk levels.

**Try it:** https://GhadahAlmaawy.github.io/maternal-health-risk-classification/

> This runs a 25-tree Random Forest directly in the browser (no server needed), trained on the same public dataset. See a note in the repo if you'd like details on how it was built.

## Tech Stack

- Python
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — preprocessing, KNN, Naive Bayes, Random Forest, Decision Tree, SVM, evaluation metrics
- imbalanced-learn — SMOTE for class balancing
- TensorFlow / Keras — Artificial Neural Network
- pydotplus — decision tree visualization

## Running the Project

```bash
git clone <this-repo-url>
cd maternal-health-risk-classification
pip install -r requirements.txt
jupyter notebook maternal_health_risk_classification.ipynb
```

> Note: the dataset (`Maternal Health Risk Data Set.csv`) is publicly available on [Kaggle](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data/data) and is not redistributed here — download it and place it in the project root to run the notebook end-to-end.

##  Team & Contribution

This was a group project completed as part of the DS312 Data Mining course. My role focused on **preprocessing , class imbalance handling, model building & evaluation**.

##  License

This project is shared for educational and portfolio purposes. purposes.
