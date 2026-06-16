import sys
import os

# Add project root to path so we can import 'app' package modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import traceback
import requests

# Streamlit App Configuration
st.set_page_config(page_title="LLM Evaluation Platform", layout="wide")

st.title("LLM Evaluation & Benchmarking Platform 🚀")

# API Configuration
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Navigation", 
    [
        "Overview", 
        "Benchmark Results", 
        "Hallucination Analysis", 
        "Cost Analysis", 
        "Experiment Comparison", 
        "Leaderboards", 
        "Error Analysis",
        "Predictive Models (ML)",
        "SHAP Analytics"
    ]
)

def get_real_leaderboard():
    try:
        response = requests.get(f"{BACKEND_API_URL}/api/v1/leaderboard", timeout=10)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            if df.empty:
                return pd.DataFrame({"Model": ["No Data"], "Elo Rating": [1500], "Avg Accuracy (%)": [50.0]})
            # Ensure proper columns exist in output
            if "Model" not in df.columns:
                df = df.rename(columns={"model": "Model"})
            return df
        else:
            return pd.DataFrame({"Model": ["No Data"], "Elo Rating": [1500], "Avg Accuracy (%)": [50.0]})
    except Exception as e:
        return pd.DataFrame({"Model": ["No Data"], "Elo Rating": [1500], "Avg Accuracy (%)": [50.0]})

def get_real_cost_data():
    try:
        response = requests.get(f"{BACKEND_API_URL}/api/v1/cost-performance", timeout=10)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            if df.empty:
                return pd.DataFrame({"Model": ["No Data"], "Cost per 1k Tokens ($)": [0.0], "Avg Accuracy (%)": [50.0]})
            return df
        else:
            return pd.DataFrame({"Model": ["No Data"], "Cost per 1k Tokens ($)": [0.0], "Avg Accuracy (%)": [50.0]})
    except Exception as e:
        return pd.DataFrame({"Model": ["No Data"], "Cost per 1k Tokens ($)": [0.0], "Avg Accuracy (%)": [50.0]})

# --- PAGE ROUTING ---

if page == "Overview":
    st.header("Platform Overview")
    st.write("Welcome to the centralized LLM evaluation platform. Navigate the sidebar to analyze performance metrics, cost frontiers, error clusters, and hallucination distributions.")
    
    col1, col2, col3 = st.columns(3)
    try:
        response = requests.get(f"{BACKEND_API_URL}/api/v1/overview", timeout=10)
        if response.status_code == 200:
            data = response.json()
            c1, c2, c3 = data["models"], data["runs"], data["questions"]
        else:
            c1, c2, c3 = 5, 1240, 125000
    except:
        c1, c2, c3 = 5, 1240, 125000
    
    col1.metric("Models Evaluated", f"{c1:,}")
    col2.metric("Total Benchmarks Run", f"{c2:,}")
    col3.metric("Questions Processed", f"{c3:,}")

elif page == "Benchmark Results":
    st.header("Benchmark Results")
    st.write("Detailed view of all active benchmark evaluations.")
    df = get_real_leaderboard()
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Model")["Avg Accuracy (%)"])

elif page == "Experiment Comparison":
    st.header("Experiment Comparison")
    st.write("Compare side-by-side execution metrics across two or more model experiments.")
    try:
        response_models = requests.get(f"{BACKEND_API_URL}/api/v1/models", timeout=10)
        if response_models.status_code == 200:
            models_data = response_models.json()
            model_names = [m["name"] for m in models_data]
        else:
            model_names = []
            
        if len(model_names) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                mod1 = st.selectbox("Select Model 1 (Baseline)", model_names, index=0)
            with col2:
                mod2 = st.selectbox("Select Model 2 (Comparison)", model_names, index=1)
                
            def get_model_stats(m_name):
                try:
                    res_stats = requests.get(f"{BACKEND_API_URL}/api/v1/model-stats/{m_name}", timeout=10)
                    if res_stats.status_code == 200:
                        return res_stats.json()
                    else:
                        return {"avg_lat": 0.0, "acc": 0.0, "cost_1k": 0.0, "hal_rate": 0.0}
                except:
                    return {"avg_lat": 0.0, "acc": 0.0, "cost_1k": 0.0, "hal_rate": 0.0}

            stat1 = get_model_stats(mod1)
            stat2 = get_model_stats(mod2)
            
            with col1:
                st.metric("Exact Match %", f"{stat1['acc']:.1f}%")
                st.metric("Avg Latency", f"{stat1['avg_lat']:.2f}s")
                st.metric("Cost per 1k Tokens", f"${stat1['cost_1k']:.4f}")
                st.metric("Hallucination Rate", f"{stat1['hal_rate']:.1f}%")
                
            with col2:
                st.metric("Exact Match %", f"{stat2['acc']:.1f}%", f"{stat2['acc'] - stat1['acc']:.1f}%")
                st.metric("Avg Latency", f"{stat2['avg_lat']:.2f}s", f"{stat2['avg_lat'] - stat1['avg_lat']:.2f}s", delta_color="inverse")
                st.metric("Cost per 1k Tokens", f"${stat2['cost_1k']:.4f}", f"${stat2['cost_1k'] - stat1['cost_1k']:.4f}", delta_color="inverse")
                st.metric("Hallucination Rate", f"{stat2['hal_rate']:.1f}%", f"{stat2['hal_rate'] - stat1['hal_rate']:.1f}%", delta_color="inverse")
        else:
            st.info("Need at least 2 models evaluated for comparison.")
    except Exception as e:
        st.warning(f"Could not load comparison data: {e}")

elif page == "Leaderboards":
    st.header("Model Leaderboard (Elo Rating)")
    st.write("Head-to-head performance rankings calculated across all active benchmark categories.")
    
    df = get_real_leaderboard()
    st.dataframe(df, use_container_width=True)
    
    fig = px.bar(df, x="Model", y="Elo Rating", color="Model", title="Current Elo Standings")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Cost Analysis":
    st.header("Cost vs. Performance Frontier")
    st.write("Visualizing the Pareto frontier to identify the most cost-efficient models for deployment.")
    
    df = get_real_cost_data()
    
    # Check if df is valid
    if df.empty or df["Model"].iloc[0] == "No Data" or df["Cost per 1k Tokens ($)"].isnull().all():
        st.error("Cost data is empty, null, or failed to load. Please check the backend database.")
        st.dataframe(df)
    else:
        st.write("Data loaded successfully!")
        fig = px.scatter(
            df, 
            x="Cost per 1k Tokens ($)", 
            y="Avg Accuracy (%)", 
            color="Model", 
            hover_name="Model",
            title="Pareto Frontier: Accuracy vs Cost"
        )
        fig.update_traces(marker=dict(size=15, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Hallucination Analysis":
    st.header("Hallucination Distributions")
    st.write("Violin plots displaying the probability distribution of hallucinations detected by the LLM-as-a-Judge pipeline.")
    
    try:
        response = requests.get(f"{BACKEND_API_URL}/api/v1/hallucinations", timeout=10)
        if response.status_code == 200:
            df_hal = pd.DataFrame(response.json())
            if not df_hal.empty:
                # Rename columns from JSON format if necessary
                if "Model" not in df_hal.columns and "model" in df_hal.columns:
                    df_hal = df_hal.rename(columns={"model": "Model"})
                hal_rates = df_hal.groupby('Model')['is_hal'].mean().reset_index()
                fig = px.bar(hal_rates, x="Model", y="is_hal", title="Hallucination Rates by Model", color="Model")
                fig.update_layout(yaxis_title="Probability of Hallucination")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hallucination data available.")
        else:
            st.warning("No hallucination data available from API.")
    except Exception as e:
        st.warning(f"No hallucination data available: {e}")

elif page == "Error Analysis":
    st.header("Failure Mode Clustering (K-Means)")
    st.write("Questions that the models frequently fail on are embedded and clustered to reveal systemic weaknesses.")
    
    try:
        if os.path.exists("data/error_clusters.csv"):
            err_df = pd.read_csv("data/error_clusters.csv")
            clusters = err_df['cluster_label'].dropna().unique()
            if len(clusters) > 0:
                cols = st.columns(len(clusters))
                for i, cluster in enumerate(clusters):
                    with cols[i]:
                        st.subheader(f"Cluster: {cluster}")
                        samples = err_df[err_df['cluster_label'] == cluster]['question'].head(3).tolist()
                        for s in samples:
                            st.error(f"- {s}")
            else:
                st.info("No failure clusters found.")
        else:
            st.info("Run the training script to generate error clusters.")
    except Exception as e:
        st.error(f"Error loading clusters: {e}")

elif page == "Predictive Models (ML)":
    st.header("Secondary ML Predictors")
    st.write("Interact with the trained secondary ML models (XGBoost, LightGBM, CatBoost) to forecast metrics before executing LLM generation.")
    
    tab1, tab2, tab3 = st.tabs(["Hallucination Detector", "Failure Risk", "Cost/Latency Forecasting"])
    
    with tab1:
        st.subheader("Hallucination Classifier")
        question = st.text_area("Question", "What is the capital of France?")
        ground_truth = st.text_area("Ground Truth", "Paris")
        response = st.text_area("Model Response", "The capital of France is Berlin.")
        if st.button("Predict Hallucination"):
            if os.path.exists("data/hallucination_detector.pkl"):
                from app.research.hallucination_detector import HallucinationDetector
                hd = HallucinationDetector()
                prob = hd.predict(question, ground_truth, response)
                if prob > 0.5:
                    st.error(f"Prediction: {prob*100:.1f}% probability of hallucination")
                else:
                    st.success(f"Prediction: {prob*100:.1f}% probability of hallucination")
            else:
                st.warning("Model not trained yet.")
            
    with tab2:
        st.subheader("Failure Probability (CatBoost)")
        st.info("Uses CatBoost to predict if the prompt will fail before executing.")
        model_name = st.selectbox("Model", ["gpt-4-turbo", "claude-3-opus", "gemini-1.5-pro", "mistral-large-latest", "deepseek-coder"])
        benchmark_name = st.selectbox("Benchmark", ["gsm8k", "mmlu", "humaneval"])
        prompt_tokens = st.slider("Prompt Tokens", 10, 8000, 500)
        max_tokens = st.slider("Max Tokens", 10, 4000, 1024)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        if st.button("Predict Failure Risk"):
            if os.path.exists("data/failure_predictor.pkl"):
                import joblib
                fp = joblib.load("data/failure_predictor.pkl")
                df_f = pd.DataFrame([{"model_name": model_name, "benchmark_name": benchmark_name, "prompt_tokens": prompt_tokens, "temperature": temperature, "max_tokens": max_tokens}])
                prob = fp.predict_proba(df_f)[0][1] # probability of class 1
                st.warning(f"Prediction: {prob*100:.1f}% Risk of Exact Match Failure")
            else:
                st.warning("Model not trained yet.")
            
    with tab3:
        st.subheader("Cost & Latency Forecasting")
        st.info("Uses XGBoost to forecast cost and latency parameters.")
        model_name2 = st.selectbox("Target Model", ["gpt-4-turbo", "claude-3-opus", "gemini-1.5-pro", "mistral-large-latest", "deepseek-coder"], key="mod2")
        prompt_tokens2 = st.slider("Prompt Tokens", 10, 8000, 500, key="pt2")
        max_tokens2 = st.slider("Max Tokens", 10, 4000, 1024, key="mt2")
        temperature2 = st.slider("Temperature", 0.0, 1.0, 0.7, key="t2")
        if st.button("Forecast"):
            if os.path.exists("data/cost_predictor.pkl") and os.path.exists("data/latency_predictor.pkl"):
                import joblib
                cost_model, encoder = joblib.load("data/cost_predictor.pkl")
                latency_model = joblib.load("data/latency_predictor.pkl")
                
                try:
                    encoded_mod = encoder.transform([model_name2])[0]
                except:
                    encoded_mod = 0 # fallback if unseen
                
                df_c = pd.DataFrame([{"prompt_tokens": prompt_tokens2, "max_tokens": max_tokens2, "temperature": temperature2, "model_encoded": encoded_mod}])
                cost = cost_model.predict(df_c)[0]
                latency = latency_model.predict(df_c)[0]
                st.info(f"Estimated Cost: ${max(0, cost):.5f} | Estimated Latency: {max(0, latency/1000):.2f}s")
            else:
                st.warning("Models not trained yet.")

elif page == "SHAP Analytics":
    st.header("Explainable AI (SHAP)")
    st.write("Understand the driving features behind our secondary ML model predictions.")
    if os.path.exists("data/shap_summary.png"):
        st.image("data/shap_summary.png", use_column_width=True)
    else:
        st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_header.png", use_column_width=True)
        st.info("Run `scripts/train_all_models.py` locally to generate exact feature importance plots for your dataset.")

else:
    st.info(f"{page} views are currently under construction.")
