import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.research.data_pipeline import fetch_training_data
from app.research.hallucination_detector import HallucinationDetector
from app.research.llm_judge import LLMJudge
from app.research.failure_predictor import FailurePredictor
from app.research.cost_latency_predictors import CostLatencyPredictors
from app.research.error_analysis import ErrorAnalyzer
from app.research.explainability import SHAPExplainer

def main():
    print("="*50)
    print("PHASE 2: TRAINING & RESEARCH MODULE ORCHESTRATOR")
    print("="*50)
    
    # 1. Fetch Data
    print("\n[1] Fetching training data from PostgreSQL...")
    try:
        df = fetch_training_data()
        print(f"Successfully loaded {len(df)} records.")
    except Exception as e:
        print(e)
        return

    # 2. Train Hallucination Detector
    print("\n[2] Training Hallucination Detector...")
    hd = HallucinationDetector()
    hd.train(df)
    
    # 3. Train LLM Judge
    print("\n[3] Training LLM Judge...")
    judge = LLMJudge()
    judge.train(df)
    
    # 4. Train Failure Predictor
    print("\n[4] Training Failure Predictor...")
    fp = FailurePredictor()
    fp.train(df)
    
    # 5. Train Cost & Latency Predictors
    print("\n[5] Training Cost & Latency Predictors...")
    cl = CostLatencyPredictors()
    cl.train(df)
    
    # 6. Error Analysis (Unsupervised)
    print("\n[6] Running Error Analysis (UMAP + KMeans)...")
    ea = ErrorAnalyzer()
    failed_df = ea.analyze(df)
    failed_df.to_csv("data/error_clusters.csv", index=False)
    print("Saved error clusters to data/error_clusters.csv")
    
    # 7. Model Explainability (SHAP)
    print("\n[7] Generating SHAP Explanations...")
    shap_exp = SHAPExplainer()
    # We will explain the failure predictor
    features = ['model_name', 'benchmark_name', 'prompt_tokens', 'temperature', 'max_tokens']
    X_fp = df[features].copy()
    X_fp['model_name'] = X_fp['model_name'].astype(str)
    X_fp['benchmark_name'] = X_fp['benchmark_name'].astype(str)
    
    explainer, shap_values = shap_exp.explain_failure_predictor(fp.classifier, X_fp)
    shap_exp.plot_feature_importance(explainer, shap_values, X_fp)

    print("\n" + "="*50)
    print("ALL ML MODELS TRAINED AND SAVED SUCCESSFULLY.")
    print("="*50)

if __name__ == "__main__":
    main()
