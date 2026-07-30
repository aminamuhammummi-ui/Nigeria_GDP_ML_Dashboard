# Nigeria GDP Growth – Machine Learning Dashboard

**A Machine Learning Approach to Modelling the Determinants of Economic Growth in Nigeria**

This interactive Streamlit dashboard presents the results of a supervised machine learning study that predicts Nigeria’s annual GDP growth using key macroeconomic indicators. The study compares traditional Ordinary Least Squares (OLS) regression with modern machine learning models (Random Forest, Gradient Boosting, and Neural Network).

---

## Live Demo

🔗 **[Click here to open the live dashboard](https://nigeriagdpmldashboard-jsec9nnctqelkjhpvj4w6z.streamlit.app/)**  

---

## Google Colab Notebooks

The complete machine learning workflow is documented in two Google Colab notebooks:

- **Phase 1 – Data Collection & Preprocessing**  
  🔗 https://colab.research.google.com/drive/1BHSFzAVQF9VRj6hwaHchDVVkVQE4-yuc?usp=sharing 

- **Phase 2 – Machine Learning Modelling & Evaluation**  
  🔗 https://colab.research.google.com/drive/1ni9wqpbMf28gAFcZ795fu5uL0Gzgcg8E?usp=sharing 

These notebooks provide the full workflow used in this project, including data acquisition from the World Bank, preprocessing, feature engineering, model training, evaluation, and export of the final models used in the dashboard.

---

## Project Overview

- **Aim**: Develop and evaluate supervised machine learning models for predicting Nigeria’s annual GDP growth and compare their performance against a traditional OLS benchmark.
- **Best Model**: Gradient Boosting (lowest RMSE on the test set)
- **Data Period**: 1990 – 2025 (annual)
- **Data Source**: World Bank – World Development Indicators (WDI)

### Key Finding
Ensemble machine learning models (especially Gradient Boosting and Random Forest) significantly outperform the traditional OLS regression model in predictive accuracy.

---

## Features of the Dashboard

1. **Project Overview** – Summary of the study and key results
2. **Data & Variables** – Full transparency on variables used, variables excluded, and justification linked to the research aim and objectives
3. **Historical Trends** – Visualisation of GDP growth and key macroeconomic indicators
4. **Model Performance** – Comparison of OLS, Random Forest, Gradient Boosting, and Neural Network
5. **Feature Importance** – Which variables contributed most to the predictions
6. **What-If Scenario Simulator** – Interactive tool to simulate policy scenarios and see predicted GDP growth

---

## Variables Used

| Variable                        | Description                                      | Role                  |
|--------------------------------|--------------------------------------------------|-----------------------|
| GDP growth (annual %)          | Target variable                                  | Dependent             |
| FDI (% of GDP)                 | Foreign Direct Investment net inflows            | Independent           |
| Inflation (CPI, annual %)      | Consumer price inflation                         | Independent           |
| Population growth (annual %)   | Annual population growth rate                    | Independent           |
| Internet users (% of population)| Proxy for digital infrastructure / human capital | Independent           |
| Lagged variables               | Previous year’s GDP growth, inflation, and FDI   | Independent           |

**Note**: Gross capital formation, government expenditure, and trade openness were originally planned but could not be included because the World Bank WDI returned no data for these series for Nigeria during the study period.

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/aminamuhammummi-ui/Nigeria_GDP_ML_Dashboard.git
cd Nigeria_GDP_ML_Dashboard
```

### 2. Clone the repository

```bash
python -m venv venv

# Activate it
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

```

### 3. Install dependencies
```bash
pip install -r requirements.txt

```
#### OR
```bash
pip install --upgrade pip 
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
```

### 4. Run the app
```bash
streamlit run app.py

```