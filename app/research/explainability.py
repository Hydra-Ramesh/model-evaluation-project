import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self):
        pass
        
    def explain_failure_predictor(self, model, X: pd.DataFrame):
        """
        Generates SHAP values for the CatBoost failure predictor.
        """
        print("Generating SHAP Explanations...")
        # For CatBoost, shap.TreeExplainer works perfectly
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Return SHAP values and expected value for dashboard visualization
        return explainer, shap_values
        
    def plot_feature_importance(self, explainer, shap_values, X, output_path="data/shap_summary.png"):
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP summary to {output_path}")
