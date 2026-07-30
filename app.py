import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Nigeria GDP Growth ML Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------
# Load data & models
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_modelling_data.csv")
    metrics = pd.read_csv("model_metrics.csv")
    preds = pd.read_csv("test_predictions.csv")
    return df, metrics, preds

@st.cache_resource
def load_models():
    models = {
        "OLS": joblib.load("model_ols.joblib"),
        "Random Forest": joblib.load("model_rf.joblib"),
        "Gradient Boosting": joblib.load("model_gb.joblib"),
        "Neural Network": joblib.load("model_nn.joblib")
    }
    return models

df, metrics, preds = load_data()
models = load_models()

features = ['fdi', 'inflation', 'pop_growth', 'internet_users',
            'gdp_growth_lag1', 'inflation_lag1', 'fdi_lag1']

# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "1. Project Overview",
    "2. Historical Trends",
    "3. Data & Variables",
    "4. Model Performance",
    "5. Feature Importance",
    "6. What-If Scenario Simulator"
])

st.title("Nigeria Economic Growth – Machine Learning Dashboard")
st.caption("A Machine Learning Approach to Modelling the Determinants of Economic Growth in Nigeria")

# --------------------------
# Page 1: Overview
# --------------------------
if page == "1. Project Overview":
    st.header("Project Overview")
    st.markdown("""
    This interactive dashboard presents the results of a supervised machine learning study 
    that predicts **Nigeria’s annual GDP growth** using key macroeconomic indicators.
    
    **Models compared:**
    - Ordinary Least Squares (OLS) – traditional econometric benchmark
    - Random Forest
    - Gradient Boosting ← **Best performing model**
    - Neural Network (MLP)
    
    **Main finding:**  
    Ensemble machine learning models significantly outperform the traditional OLS regression.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Data Period", f"{df['year'].min()} – {df['year'].max()}")
    col2.metric("Best Model", "Gradient Boosting")
    col3.metric("Best RMSE", f"{metrics.loc[metrics['Model']=='Gradient Boosting', 'RMSE'].values[0]}")

    st.subheader("Predictors Used")
    st.write(", ".join(features))

# --------------------------
# Page 2: Historical Trends
# --------------------------
elif page == "2. Historical Trends":
    st.header("Historical Trends")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df['year'], df['gdp_growth'], marker='o', linewidth=2, color='#1f4e79')
    ax.axhline(0, color='red', linestyle='--', alpha=0.6)
    ax.set_title("Nigeria – Annual GDP Growth (%)", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP Growth (%)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Inflation Rate")
        st.line_chart(df.set_index('year')['inflation'], color="#e67e22")
    with col2:
        st.subheader("Foreign Direct Investment (% of GDP)")
        st.line_chart(df.set_index('year')['fdi'], color="#27ae60")


# --------------------------
# Page 3: Data & Variables
# --------------------------
elif page == "3. Data & Variables":
    st.header("Data & Variables Used in the Study")

    st.markdown("""
    This section provides full transparency about the dataset used for modelling. 
    It explains the data source, the variables that were finally included, the variables 
    that could not be included, and — most importantly — **why each selected variable 
    is relevant to the aim, scope, and objectives of this study**.
    """)

    # -------------------------------------------------------
    st.subheader("1. Data Source")
    st.markdown("""
    - **Primary Source**: World Bank – World Development Indicators (WDI)  
    - **Country**: Nigeria (`NGA`)  
    - **Frequency**: Annual  
    - **Period**: 1990 – 2025 (latest available at the time of extraction)  
    - **Access Method**: World Bank API via the `wbgapi` Python package
    """)

    # -------------------------------------------------------
    st.subheader("2. Alignment with Project Aim, Scope and Objectives")
    st.markdown("""
    **Aim of the Study**  
    To develop and compare supervised machine learning regression models for predicting 
    Nigeria’s annual GDP growth rates, and to benchmark their predictive performance 
    against a conventional Ordinary Least Squares (OLS) model.

    **Scope**  
    The study is limited to Nigeria and uses annual macroeconomic time-series data. 
    The focus is on modelling the key macroeconomic determinants of economic growth 
    using data-driven methods that can capture non-linear relationships.

    The variables selected below were chosen because they are:
    - Theoretically grounded in growth theory (Neoclassical and Endogenous Growth Theory)
    - Empirically supported in studies of developing economies and Nigeria
    - Consistent with the project objectives of building a structured dataset and training 
      supervised regression models for GDP growth prediction
    """)

    # -------------------------------------------------------
    st.subheader("3. Variables Finally Used and Their Justification")

    st.markdown("#### Dependent Variable")
    st.markdown("""
    | Variable | Justification |
    |----------|---------------|
    | **GDP growth (annual %)** | This is the official and most widely used measure of economic growth. It directly represents the outcome the study aims to predict and is the standard dependent variable in both traditional econometric and machine learning growth models. |
    """)

    st.markdown("#### Independent Variables")
    st.markdown("""
    | Variable | Theoretical & Empirical Justification | Relevance to Project Objectives |
    |----------|---------------------------------------|---------------------------------|
    | **Foreign Direct Investment (FDI) net inflows (% of GDP)** | FDI brings capital, technology transfer, and productivity spillovers (Borensztein et al., 1998). It is especially important for developing economies such as Nigeria. | Supports Objective 2 (dataset construction) and Objective 3 (training ML models) by capturing external capital inflows that influence growth. |
    | **Inflation, consumer prices (annual %)** | High and volatile inflation distorts investment decisions and reduces real returns, negatively affecting growth (supported by Nigerian studies such as Orji et al., 2018). | Critical for modelling the macroeconomic instability that characterises the Nigerian economy — a key motivation of the study. |
    | **Population growth (annual %)** | In the neoclassical growth model (Solow-Swan), population growth affects the capital–labour ratio and long-run output per worker. | Included to capture demographic dynamics that influence labour supply and per capita growth outcomes. |
    | **Individuals using the Internet (% of population)** | Serves as a proxy for digital infrastructure and human capital development. Endogenous growth theory (Romer, Lucas) emphasises knowledge and technology as drivers of sustained growth. | Addresses the “human capital / digital measures” mentioned in the project scope and strengthens the model’s ability to capture modern growth drivers. |
    | **Lagged GDP growth, Inflation, and FDI** | Macroeconomic effects often operate with a delay. Including one-year lags allows the models to capture persistence and dynamic relationships in the time series. | Improves the predictive power of the supervised models (Objective 3) and reflects standard practice in time-series growth modelling. |
    """)

    # -------------------------------------------------------
    st.subheader("4. Variables Originally Planned but Not Used")

    missing_data = {
        "Variable": [
            "Gross capital formation (% of GDP)",
            "General government final consumption expenditure (% of GDP)",
            "Trade (% of GDP)",
            "Exports of goods and services (% of GDP)",
            "Imports of goods and services (% of GDP)"
        ],
        "WDI Code": [
            "NE.GDI.TOTL.ZS / NE.GDI.FTOT.ZS",
            "NE.CON.GOVT.ZS",
            "NE.TRD.GNFS.ZS",
            "NE.EXP.GNFS.ZS",
            "NE.IMP.GNFS.ZS"
        ],
        "Reason for Exclusion": [
            "No data available for Nigeria in the World Development Indicators for the entire study period.",
            "No data available for Nigeria in WDI for the study period.",
            "No data available for Nigeria in WDI for the study period.",
            "No data available for Nigeria in WDI for the study period.",
            "No data available for Nigeria in WDI for the study period."
        ]
    }
    st.dataframe(pd.DataFrame(missing_data), use_container_width=True, hide_index=True)

    st.markdown("""
    **Why these variables were originally planned**  
    - **Gross capital formation** is a core variable in both neoclassical and endogenous growth theory (physical capital accumulation).  
    - **Government expenditure** captures the role of fiscal policy.  
    - **Trade openness** is widely linked to technology diffusion and efficiency gains in developing countries.

    Despite their theoretical importance, these series returned completely missing values 
    for Nigeria across the full 1990–2025 period when accessed via both the World Bank 
    DataBank and the official API. This is a recognised data coverage limitation in the 
    World Development Indicators for certain national accounts series.
    """)

    # -------------------------------------------------------
    st.subheader("5. Final Modelling Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Observations Used", len(df))
    col2.metric("Time Period", f"{df['year'].min()} – {df['year'].max()}")
    col3.metric("Features in Final Models", len(features))

    st.info("""
    **Note for users and examiners**:  
    The final set of variables remains theoretically grounded and consistent with the 
    project aim of testing whether machine learning methods can improve upon traditional 
    OLS models in predicting Nigerian GDP growth, even under data constraints.
    """)

# --------------------------
# Page 4: Model Performance
# --------------------------
elif page == "4. Model Performance":
    st.header("Model Performance Comparison (Test Set)")

    st.dataframe(
        metrics.style
        .highlight_min(subset=['RMSE', 'MAE'], color='#d4edda')
        .highlight_max(subset=['R²'], color='#d4edda'),
        use_container_width=True
    )

    st.subheader("Actual vs Predicted GDP Growth")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(preds['year'], preds['actual'], 'ko-', label='Actual', linewidth=2.5)

    colors = {
        'OLS': '#e74c3c',
        'Random Forest': '#3498db',
        'Gradient Boosting': '#2ecc71',
        'Neural Network': '#9b59b6'
    }
    for col in ['OLS', 'Random Forest', 'Gradient Boosting', 'Neural Network']:
        if col in preds.columns:
            ax.plot(preds['year'], preds[col], marker='o', label=col, color=colors[col], alpha=0.85)

    ax.set_title("Actual vs Predicted (Hold-out Test Period)")
    ax.set_ylabel("GDP Growth (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='grey', linestyle='--')
    st.pyplot(fig)

    st.success("Gradient Boosting achieved the lowest RMSE and is selected as the primary model for policy simulation.")

# --------------------------
# Page 5: Feature Importance
# --------------------------
elif page == "5. Feature Importance":
    st.header("Feature Importance")

    rf_imp = pd.Series(models['Random Forest'].feature_importances_, index=features).sort_values()
    gb_imp = pd.Series(models['Gradient Boosting'].feature_importances_, index=features).sort_values()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Random Forest")
        fig, ax = plt.subplots(figsize=(6, 4))
        rf_imp.plot.barh(ax=ax, color='#3498db')
        ax.set_xlabel("Importance")
        st.pyplot(fig)

    with col2:
        st.subheader("Gradient Boosting")
        fig, ax = plt.subplots(figsize=(6, 4))
        gb_imp.plot.barh(ax=ax, color='#e67e22')
        ax.set_xlabel("Importance")
        st.pyplot(fig)

# --------------------------
# Page 6: What-If Simulator
# --------------------------
elif page == "6. What-If Scenario Simulator":
    st.header("What-If Policy Scenario Simulator")
    st.write("Adjust the values below to simulate different macroeconomic conditions. Predictions are generated by the **Gradient Boosting** model.")

    col1, col2, col3 = st.columns(3)

    with col1:
        fdi = st.slider("FDI (% of GDP)", -1.0, 5.0, 1.0, 0.1)
        inflation = st.slider("Inflation (%)", 0.0, 40.0, 15.0, 0.5)
    with col2:
        pop_growth = st.slider("Population Growth (%)", 1.5, 3.5, 2.3, 0.05)
        internet = st.slider("Internet Users (% of population)", 0.0, 55.0, 35.0, 1.0)
    with col3:
        gdp_lag = st.slider("Previous Year GDP Growth (%)", -8.0, 12.0, 3.0, 0.1)
        infl_lag = st.slider("Previous Year Inflation (%)", 0.0, 40.0, 15.0, 0.5)
        fdi_lag = st.slider("Previous Year FDI (%)", -1.0, 5.0, 1.0, 0.1)

    input_data = pd.DataFrame(
        [[fdi, inflation, pop_growth, internet, gdp_lag, infl_lag, fdi_lag]],
        columns=features
    )

    prediction = models['Gradient Boosting'].predict(input_data)[0]

    st.markdown("---")
    st.subheader("Predicted Annual GDP Growth")
    
    # Big metric display
    st.metric(label="Predicted GDP Growth", value=f"{prediction:.2f}%")

    if prediction >= 4:
        st.success("Strong positive growth expected")
    elif prediction >= 2:
        st.info("Moderate growth expected")
    elif prediction >= 0:
        st.warning("Low but positive growth")
    else:
        st.error("Negative growth predicted")