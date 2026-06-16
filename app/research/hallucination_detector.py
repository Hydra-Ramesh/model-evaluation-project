import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sentence_transformers import SentenceTransformer

class HallucinationDetector:
    def __init__(self, model_path="data/hallucination_detector.pkl"):
        self.model_path = model_path
        # We use a lightweight transformer for text feature extraction before feeding to XGBoost
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = None
        
    def train(self, df: pd.DataFrame):
        print("Training Hallucination Detector (XGBoost)...")
        # Feature Engineering: Combine question, ground truth, and response
        texts = df['question'] + " [SEP] " + df['ground_truth'] + " [SEP] " + df['model_response']
        print("Generating embeddings (this may take a moment)...")
        X = self.embedder.encode(texts.tolist(), show_progress_bar=True)
        y = df['hallucination_detected'].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)
        self.classifier.fit(X_train, y_train)
        
        preds = self.classifier.predict(X_test)
        probs = self.classifier.predict_proba(X_test)[:, 1]
        
        print(f"Hallucination Model - Accuracy: {accuracy_score(y_test, preds):.4f}")
        print(f"Hallucination Model - F1 Score: {f1_score(y_test, preds):.4f}")
        print(f"Hallucination Model - ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.classifier, self.model_path)
        print(f"Saved to {self.model_path}")
        
    def predict(self, question: str, ground_truth: str, response: str) -> float:
        if self.classifier is None:
            self.classifier = joblib.load(self.model_path)
        
        text = question + " [SEP] " + ground_truth + " [SEP] " + response
        emb = self.embedder.encode([text])
        prob = self.classifier.predict_proba(emb)[0][1]
        return float(prob)
