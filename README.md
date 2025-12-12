NHANES Metabolic Health Analysis
This repo contains all my data, cleaning scripts, and eda for Sprint 2 of my INST414 Capstone project. MY goal is to explore metabolic health indicators and find its risk factors using NHANES data and prepare a cleaned dataset ot get ready for sprint 3 models.



<img width="811" height="860" alt="image" src="https://github.com/user-attachments/assets/8a092c40-144d-4d41-9716-8eb7955e03d1" />


Data Sources
NHANES datasets were downloaded from the CDC and kaggle
 https://wwwn.cdc.gov/nchs/nhanes/
National Health and Nutrition Examination Survey
This project uses the following NHANES data:
Demographics


Diet


Examination


Labs


Medications


Questionnaire

📦 Data Structure
data/
├── demographic.csv
├── diet.csv
├── examination.csv
├── labs.csv
├── medications.csv
└── questionnaire.csv

clean_data/
└── cleaned_data_safe.zip     # Cleaned dataset (zipped because CSV was too large)

scripts/
└── clean.py                  # Data cleaning & merging script

notebooks/
└── eda.ipynb                 # Exploratory Data Analysis with visuals

README.md                     # Sprint 2 documentation

📂 Data Sources

NHANES datasets were downloaded from:

CDC — https://www.cdc.gov/nchs/nhanes/

Kaggle (optional alternative)

This project uses the following NHANES components:

Demographics

Diet

Examination

Laboratory

Medications

Questionnaire

📝 Project Summary (Sprint 2)

This repository contains the raw NHANES datasets, cleaning scripts, and exploratory data analysis (EDA) used to prepare a cleaned dataset for Sprint 2 of my INST414 Capstone.
My goal is to explore metabolic health indicators and prepare the merged, cleaned dataset for Sprint 3 modeling.

🚀 Contents

data/
Original NHANES CSVs.

clean_data/
Final cleaned dataset (zipped due to GitHub size limits).

scripts/
clean.py — merges, cleans, and filters NHANES components.

notebooks/
eda.ipynb — visualizations, distributions, missing data checks.
