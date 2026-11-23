# Heart Disease Prediction – Comparing Machine Learning Models

Task 02 of the **Devixo Solutions AI/ML Internship**. This project trains six machine learning classifiers to predict whether a patient has heart disease from clinical measurements, then compares them to pick the most dependable one.

## Dataset

- **Source:** [Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Raw size:** 1,025 patients, 14 columns
- **After removing 723 duplicate rows:** 302 unique patients
- **Target:** `target` (1 = heart disease, 0 = no heart disease)
- **Main features:** age, sex, chest-pain type (`cp`), resting blood pressure (`trestbps`), cholesterol (`chol`), max heart rate (`thalach`), ST depression (`oldpeak`), and more

## Workflow

1. **Preprocessing:** check missing values, remove duplicates, engineer `thalach_pct` (peak heart rate as a share of the age-predicted maximum), one-hot encode `cp`, `restecg`, `slope` and `thal`, stratified 80/20 split, scale numeric columns (scaler fitted on training data only)
2. **Models:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost
3. **Evaluation:** Accuracy, Precision, Recall, F1, confusion matrices, 5-fold cross-validation, training time
4. **Comparison:** table with metrics, advantages and limitations of each model
5. **Bonus:** hyperparameter tuning of Random Forest with `GridSearchCV`, plus Linear Regression to predict max heart rate (MAE, MSE, RMSE, R²)

## Results

| Model | Accuracy | Precision | Recall | F1 | CV Acc. | Train time (s) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.852** | **0.875** | 0.848 | **0.862** | 0.793 | 0.052 |
| Random Forest | 0.803 | 0.818 | 0.818 | 0.818 | 0.809 | 0.851 |
| SVM | 0.803 | 0.818 | 0.818 | 0.818 | 0.818 | 0.010 |
| Decision Tree | 0.787 | 0.778 | 0.848 | 0.812 | 0.747 | 0.013 |
| KNN | 0.787 | 0.812 | 0.788 | 0.800 | 0.817 | 0.004 |
| XGBoost | 0.754 | 0.781 | 0.758 | 0.769 | 0.817 | 1.340 |

**Best model:** Logistic Regression (highest test F1, fast, and easy to explain).

**Hyperparameter tuning:** the tuned Random Forest (`n_estimators=200`, `max_depth=8`, `min_samples_split=5`, `min_samples_leaf=1`) improved test F1 from 0.818 to 0.824 and recall from 0.818 to 0.848, but still did not beat Logistic Regression.

**Linear Regression (max heart rate):** MAE 14.34, RMSE 18.44, R² 0.271.

## Limitations

- After removing duplicates only 302 patients remain, and the test set has just 61, so one patient changes accuracy by about 1.6 points. The top models are close and the ranking could change on a different split.
- Logistic Regression's cross-validated accuracy (0.793) is lower than its test accuracy (0.852), so the result should be treated with caution.
- Predictions support, and do not replace, a doctor's diagnosis.

## How to Run

1. Open `Task_02_AI_ML.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run all cells (`Runtime` → `Run all`). The dataset downloads automatically through `kagglehub`; if that fails, upload `heart.csv` when prompted.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Google Colab

## Project Structure

```
├── Task_02_AI_ML.ipynb      # full notebook with outputs
└── Task_02_AI_ML.pdf        # report
```

## Author

**Aman Siddiqui** 
