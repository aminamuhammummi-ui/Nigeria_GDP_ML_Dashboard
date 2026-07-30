import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the data
df = pd.read_csv("final_modelling_data.csv")
print("Data shape:", df.shape)

target = 'gdp_growth'
features = ['fdi', 'inflation', 'pop_growth', 'internet_users',
            'gdp_growth_lag1', 'inflation_lag1', 'fdi_lag1']

X = df[features]
y = df[target]

# Time-series split (last 8 years as test)
test_size = 8
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# Define models
models = {
    'OLS': LinearRegression(),
    
    'Random Forest': RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
    
    'Neural Network': Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            max_iter=1500,
            early_stopping=True,
            random_state=42
        ))
    ])
}

# Train and evaluate
results = []
predictions = pd.DataFrame({
    'year': df['year'].iloc[-test_size:].values,
    'actual': y_test.values
})

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'RMSE': round(rmse, 3),
        'MAE': round(mae, 3),
        'R²': round(r2, 3)
    })
    
    predictions[name] = y_pred
    print(f"  → RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}")

results_df = pd.DataFrame(results).sort_values('RMSE')
print("\nModel Performance:")
print(results_df)

# Save everything
joblib.dump(models['OLS'], 'model_ols.joblib')
joblib.dump(models['Random Forest'], 'model_rf.joblib')
joblib.dump(models['Gradient Boosting'], 'model_gb.joblib')
joblib.dump(models['Neural Network'], 'model_nn.joblib')

predictions.to_csv('test_predictions.csv', index=False)
results_df.to_csv('model_metrics.csv', index=False)

print("\n✅ Models retrained and saved successfully with your local scikit-learn version!")