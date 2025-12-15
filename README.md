Sprint 2 Summary for NHANES Metabolic Health Analysis
This repo contains all my data, cleaning scripts, and eda for Sprint 2 of my INST414 Capstone project. MY goal is to explore metabolic health indicators and find its risk factors using NHANES data and prepare a cleaned dataset ot get ready for sprint 3 models.

 Data Structure

<img width="811" height="860" alt="image" src="https://github.com/user-attachments/assets/8a092c40-144d-4d41-9716-8eb7955e03d1" />

Data Sources

NHANES datasets were downloaded from the CDC and kaggle
 https://wwwn.cdc.gov/nchs/nhanes/
National Health and Nutrition Examination Survey
This project uses the NHANES data below:

-Demographics

-Diet

-Examination

-Labs

-Medications

-Questionnaire



Sprint 3 – Modeling Early Metabolic Dysfunction

- Added modeling.py to train three models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Uses cleaned_data_safe.csv from Sprint 2.
- Outputs:
  - Baseline vs. model metrics (accuracy, precision, recall, AUC)
  - Feature importance plots for Random Forest and XGBoost



