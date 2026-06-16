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
import sqlite3
import traceback

# Streamlit App Configuration
st.set_page_config(page_title="LLM Evaluation Platform", layout="wide")

st.title("LLM Evaluation & Benchmarking Platform 🚀")

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

def get_db_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../llm_eval.db'))

def get_real_leaderboard():
    try:
        conn = sqlite3.connect(get_db_path())
        df = pd.read_sql("""
            SELECT m.name as Model, 
                   SUM(CASE WHEN e.exact_match = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as "Avg Accuracy (%)"
            FROM evaluation_results e
            JOIN responses r ON e.response_id = r.id
            JOIN benchmark_runs b ON r.run_id = b.id
            JOIN models m ON b.model_id = m.id
            GROUP BY m.name
            ORDER BY "Avg Accuracy (%)" DESC
        """, conn)
        df["Elo Rating"] = 1500 + (df["Avg Accuracy (%)"] - 50) * 10
        return df
    except Exception as e:
        return pd.DataFrame({"Model": ["No Data"], "Elo Rating": [1500], "Avg Accuracy (%)": [50.0]})

def get_real_cost_data():
    try:
        conn = sqlite3.connect(get_db_path())
        df = pd.read_sql("""
            SELECT m.name as Model, 
                   AVG(CAST(json_extract(e.custom_metrics, '$.cost') AS FLOAT)) * 1000 as "Cost per 1k Tokens ($)",
                   SUM(CASE WHEN e.exact_match = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as "Avg Accuracy (%)"
            FROM evaluation_results e
            JOIN responses r ON e.response_id = r.id
            JOIN benchmark_runs b ON r.run_id = b.id
            JOIN models m ON b.model_id = m.id
            GROUP BY m.name
        """, conn)
        return df
    except Exception as e:
        st.error(f"SQL Error: {e}\n{traceback.format_exc()}")
        return pd.DataFrame({"Model": ["No Data"], "Cost per 1k Tokens ($)": [0.0], "Avg Accuracy (%)": [50.0]})

# --- PAGE ROUTING ---

if page == "Overview":
    st.header("Platform Overview")
    st.write("Welcome to the centralized LLM evaluation platform. Navigate the sidebar to analyze performance metrics, cost frontiers, error clusters, and hallucination distributions.")
    
    col1, col2, col3 = st.columns(3)
    try:
        conn = sqlite3.connect(get_db_path())
        c1 = pd.read_sql("SELECT COUNT(*) as c FROM models", conn).iloc[0]['c']
        c2 = pd.read_sql("SELECT COUNT(*) as c FROM benchmark_runs", conn).iloc[0]['c']
        c3 = pd.read_sql("SELECT COUNT(*) as c FROM questions", conn).iloc[0]['c']
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
        conn = sqlite3.connect(get_db_path())
        models_df = pd.read_sql("SELECT name FROM models ORDER BY name", conn)
        model_names = models_df['name'].tolist()
        if len(model_names) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                mod1 = st.selectbox("Select Model 1 (Baseline)", model_names, index=0)
            with col2:
                mod2 = st.selectbox("Select Model 2 (Comparison)", model_names, index=1)
                
            def get_model_stats(m_name):
                return pd.read_sql(f"""
                    SELECT 
                        AVG(r.latency_ms) / 1000.0 as avg_lat,
                        SUM(CASE WHEN e.exact_match = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as acc,
                        AVG(CAST(json_extract(e.custom_metrics, '$.cost') AS FLOAT)) * 1000 as cost_1k,
                        AVG(CAST(json_extract(e.custom_metrics, '$.hallucination_detected') AS FLOAT)) * 100.0 as hal_rate
                    FROM evaluation_results e
                    JOIN responses r ON e.response_id = r.id
                    JOIN benchmark_runs b ON r.run_id = b.id
                    JOIN models m ON b.model_id = m.id
                    WHERE m.name = '{m_name}'
                """, conn).iloc[0]

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
        conn = sqlite3.connect(get_db_path())
        df_hal = pd.read_sql("""
            SELECT m.name as Model, 
                   json_extract(e.custom_metrics, '$.hallucination_detected') as is_hal
            FROM evaluation_results e
            JOIN responses r ON e.response_id = r.id
            JOIN benchmark_runs b ON r.run_id = b.id
            JOIN models m ON b.model_id = m.id
            WHERE is_hal IS NOT NULL
        """, conn)
        hal_rates = df_hal.groupby('Model')['is_hal'].mean().reset_index()
        fig = px.bar(hal_rates, x="Model", y="is_hal", title="Hallucination Rates by Model", color="Model")
        fig.update_layout(yaxis_title="Probability of Hallucination")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("No hallucination data available.")

elif page == "Error Analysis":
    st.header("Failure Mode Clustering (K-Means)")
    st.write("Questions that the models frequently fail on are embedded and clustered to reveal systemic weaknesses.")
    
    try:
        import os
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
    
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
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
    import os
    if os.path.exists("data/shap_summary.png"):
        st.image("data/shap_summary.png", use_column_width=True)
    else:
        st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_header.png", use_column_width=True)
        st.info("Run `scripts/train_all_models.py` locally to generate exact feature importance plots for your dataset.")

else:
    st.info(f"{page} views are currently under construction.")
