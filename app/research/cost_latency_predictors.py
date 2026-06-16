import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

class CostLatencyPredictors:
    def __init__(self, cost_path="data/cost_predictor.pkl", latency_path="data/latency_predictor.pkl"):
        self.cost_path = cost_path
        self.latency_path = latency_path
        self.cost_model = None
        self.latency_model = None
        self.encoder = LabelEncoder()
        
    def train(self, df: pd.DataFrame):
        print("Training Cost & Latency Predictors (XGBoost)...")
        features = ['prompt_tokens', 'max_tokens', 'temperature']
        
        X = df[features].copy()
        X['model_encoded'] = self.encoder.fit_transform(df['model_name'])
        
        y_cost = df['cost']
        y_latency = df['latency_ms']
        
        X_train, X_test, yc_train, yc_test, yl_train, yl_test = train_test_split(
            X, y_cost, y_latency, test_size=0.2, random_state=42
        )
        
        self.cost_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        self.cost_model.fit(X_train, yc_train)
        pc_preds = self.cost_model.predict(X_test)
        
        self.latency_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        self.latency_model.fit(X_train, yl_train)
        pl_preds = self.latency_model.predict(X_test)
        
        print(f"Cost Predictor - MAPE: {mean_absolute_percentage_error(yc_test, pc_preds):.4f}")
        print(f"Latency Predictor - MAE (ms): {mean_absolute_error(yl_test, pl_preds):.4f}")
        
        os.makedirs(os.path.dirname(self.cost_path), exist_ok=True)
        joblib.dump((self.cost_model, self.encoder), self.cost_path)
        joblib.dump(self.latency_model, self.latency_path)
        print("Saved Cost and Latency models.")
