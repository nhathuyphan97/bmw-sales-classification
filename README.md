# BMW Sales Classification Project

## 📌 Overview
This project applies machine learning models to classify BMW car sales as **High** or **Low**.  
The dataset includes car model, year, region, color, fuel type, transmission, and other features.

## 🎯 Goal
Identify which cars are likely to generate **High sales**, so a dealership can prioritize inventory and marketing.

## 🛠️ Methods
- Data preprocessing with `pandas` and `scikit-learn`
- Class imbalance handled with **SMOTE**
- Models compared:
  - Logistic Regression
  - Decision Tree
  - Random Forest (baseline + SMOTE + threshold tuning)
  - XGBoost (tuned with GridSearchCV)

## 📊 Results
- Logistic Regression & Decision Tree: weak performance
- Random Forest: improved with SMOTE, but still biased
- **XGBoost (tuned)** gave the best results:
  - High recall for High sales (~0.94)
  - Reasonable precision (~0.31)
  - Flexible threshold tuning for different business goals
- Key sales drivers: **Model type, Year, Region, Fuel type, Color**

## 📂 Repository Structure
bmw-sales-classification/
├── data/
│ ├── raw/ # original dataset
│ └── processed/ # cleaned/transformed data
├── notebooks/ # Jupyter notebooks
├── src/ # scripts (if any)
├── results/
│ ├── figures/ # plots
│ └── metrics/ # confusion matrices, reports
├── README.md # project summary
├── requirements.txt # dependencies
├── LICENSE # license info
└── .gitignore # ignored files

bash

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/nhathuyphan97/bmw-sales-classification.git
   cd bmw-sales-classification


2. Install dependencies:
bash
pip install -r requirements.txt

3. Open the notebook:
bash
jupyter notebook notebooks/morehouse_capstone.ipynb

📈 Next Steps

- Try probability calibration (Platt/Isotonic)
- Explore AUPRC for imbalance
- Deploy best model as a simple API or dashboard
