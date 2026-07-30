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
    This section provides full transparency about the dataset used for modelling, 
    including variables that were originally planned but could not be included.
    """)

    st.subheader("1. Data Source")
    st.markdown("""
    - **Primary Source**: World Bank – World Development Indicators (WDI)  
    - **Country**: Nigeria (`NGA`)  
    - **Frequency**: Annual  
    - **Period**: 1990 – 2025 (latest available at the time of extraction)  
    - **Access Method**: World Bank API via the `wbgapi` Python package
    """)

    st.subheader("2. Variables Finally Used in the Models")

    used_data = {
        "Variable": [
            "GDP growth (annual %)",
            "Foreign Direct Investment, net inflows (% of GDP)",
            "Inflation, consumer prices (annual %)",
            "Population growth (annual %)",
            "Individuals using the Internet (% of population)",
            "Lagged GDP growth (t-1)",
            "Lagged Inflation (t-1)",
            "Lagged FDI (t-1)"
        ],
        "Code / Description": [
            "NY.GDP.MKTP.KD.ZG (Target variable)",
            "BX.KLT.DINV.WD.GD.ZS",
            "FP.CPI.TOTL.ZG",
            "SP.POP.GROW",
            "IT.NET.USER.ZS (Digital / Human capital proxy)",
            "Engineered feature",
            "Engineered feature",
            "Engineered feature"
        ],
        "Role": [
            "Dependent variable",
            "Independent variable",
            "Independent variable",
            "Independent variable",
            "Independent variable",
            "Independent variable",
            "Independent variable",
            "Independent variable"
        ]
    }
    st.dataframe(pd.DataFrame(used_data), use_container_width=True, hide_index=True)

    st.subheader("3. Importance of the Selected Variables")

    importance_data = {
        "Variable": [
            "GDP Growth",
            "Foreign Direct Investment (FDI)",
            "Inflation",
            "Population Growth",
            "Internet Users",
            "Lagged GDP Growth",
            "Lagged Inflation",
            "Lagged FDI"
        ],
        "Importance to the Study": [
            "Serves as the dependent variable and measures Nigeria's economic performance, which the study aims to predict.",
            "Represents foreign capital inflows that can stimulate investment, technology transfer, employment, and economic growth, making it central to analysing the impact of FDI on GDP growth.",
            "Captures macroeconomic stability. Inflation influences investment decisions, purchasing power, and economic performance, making it an important predictor of GDP growth.",
            "Reflects changes in labour force and market size, both of which influence long-term economic growth and development.",
            "Acts as a proxy for digital development and human capital. Increased internet access supports innovation, productivity, business efficiency, and economic transformation.",
            "Captures the persistence of economic growth by accounting for the influence of previous year's GDP growth on current economic performance.",
            "Accounts for delayed effects of inflation, recognising that macroeconomic policies and price changes often influence growth over time.",
            "Captures the delayed impact of foreign investment, as the economic benefits of FDI may not be realised immediately."
        ],
        "Link to Research Objectives": [
            "Directly supports the objective of forecasting Nigeria's GDP growth.",
            "Supports the objective of evaluating how FDI contributes to GDP growth.",
            "Supports the objective of identifying macroeconomic factors affecting GDP growth.",
            "Supports the objective of examining demographic influences on economic performance.",
            "Supports the objective of assessing how digital development contributes to economic growth.",
            "Improves forecasting accuracy by incorporating historical economic trends.",
            "Improves model performance by accounting for dynamic inflation effects.",
            "Improves prediction by capturing delayed investment effects."
        ]
    }

    st.dataframe(
        pd.DataFrame(importance_data),
        use_container_width=True,
        hide_index=True
    )

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
            "No data available for Nigeria in the World Development Indicators for the entire study period (all values returned as missing).",
            "No data available for Nigeria in WDI for the study period.",
            "No data available for Nigeria in WDI for the study period.",
            "No data available for Nigeria in WDI for the study period.",
            "No data available for Nigeria in WDI for the study period."
        ]
    }
    st.dataframe(pd.DataFrame(missing_data), use_container_width=True, hide_index=True)

    st.subheader("5. Why These Variables Could Not Be Included")
    st.markdown("""
    Despite multiple attempts using both the World Bank DataBank interface and the 
    official API (`wbgapi`), the expenditure-side national accounts series for Nigeria 
    (gross capital formation, government consumption, and trade components) returned 
    completely missing values for the period 1990–2025.

    This is a recognised data coverage limitation in the World Development Indicators 
    for some developing countries, particularly for detailed expenditure components 
    of GDP.

    **Implication for the study**:  
    The models rely on a reduced but still theoretically relevant set of predictors 
    (FDI, inflation, population growth, internet penetration, and lagged values). 
    The absence of capital formation and government expenditure data is acknowledged 
    as a limitation and is discussed in the dissertation.
    """)

    st.subheader("6. Final Modelling Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Observations Used", len(df))
    col2.metric("Time Period", f"{df['year'].min()} – {df['year'].max()}")
    col3.metric("Features in Final Models", len(features))

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