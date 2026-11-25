# Bank Account Fraud Detection

This is our project for the Projects in Machine Learning course at the University of Haifa.

The goal is to detect fraudulent bank account applications using machine learning. Since only around 1% of applications are fraudulent, accuracy alone is not enough to evaluate our models.

## Dataset

We use the Base.csv file from the Bank Account Fraud (BAF) dataset.

The dataset contains 1,000,000 applications over 8 months.

- Months 0–4: Training
- Month 5: Validation
- Months 6–7: Final testing (not used yet)

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022).

Place `Base.csv` inside the `data/` folder.

## Data Exploration

The `data_analysis/` folder contains our EDA scripts. We explored the class distribution, missing values, feature distributions, categorical fraud rates, and correlations to understand the data before training any models.

## Stage 1 — Rare Fraud

In this stage, we investigated how to evaluate a fraud detector when fraudulent applications represent only a small percentage of the data.

We performed these experiments:

1. Never-Fraud Baseline
2. Data Preprocessing
3. Standard Logistic Regression
4. Decision Threshold Analysis
5. Balanced Logistic Regression

We compared models using precision, recall, average precision, and recall at a maximum 5% false-positive rate.

Our main finding was that changing the decision threshold had a large effect on the number of detected fraud cases. Balanced Logistic Regression increased recall at the default threshold, but did not improve the overall ranking compared with standard Logistic Regression.

All model results so far were evaluated on the validation month.

## How to Run

Install the required Python libraries:

```bash
python -m pip install -r requirements.txt
```

Run all Stage 1 experiments:

```bash
python stage1/run_all.py
```

The generated figures will be saved in `reports/stage1/`.

## Project Structure

- `data/` — Dataset (not included in GitHub)
- `data_analysis/` — EDA Python scripts
- `stage1/` — Stage 1 experiments
- `reports/` — Figures and results
- `requirements.txt` — Python dependencies

## Next Steps

Stage 2 will compare Logistic Regression with Random Forest to investigate whether nonlinear models can improve fraud detection.