# 🚀 LLM Gateway & ML Observability Platform

An enterprise-grade, intelligent API proxy and observability dashboard designed to evaluate, route, and monitor Large Language Models (LLMs) in real-time. This platform goes beyond basic telemetry by embedding **predictive Machine Learning models** directly into the routing layer to forecast failures, estimate costs, and detect hallucinations autonomously.

---

## 🌟 Core Architecture

The platform is split into two live servers that communicate asynchronously via a local SQLite data warehouse:

1. **API Gateway (`app.main:app`)**: A FastAPI server that acts as a drop-in proxy. It mimics the OpenAI API schema but intercepts requests, logs telemetry, runs ML predictions, and dynamically routes the request to any target LLM (Anthropic, Gemini, Mistral, Meta, etc.).
2. **Observability Dashboard (`app/visualization/dashboard.py`)**: A multi-tab Streamlit dashboard offering real-time visualization of LLM execution, cost Pareto frontiers, hallucination distributions, and Elo leaderboards.

---

## 📂 Repository Structure

```text
├── app/
│   ├── database/         # SQLite DB connection & SQLAlchemy Models
│   ├── evaluation/       # LLM-as-a-Judge semantic scoring & exact match metrics
│   ├── models/           # The UniversalConnector for LLM routing (LiteLLM/SDKs)
│   ├── research/         # ML pipeline: XGBoost, CatBoost, SentenceTransformers, SHAP
│   ├── visualization/    # Streamlit dashboard & Plotly charts
│   └── main.py           # FastAPI proxy server entrypoint
├── data/                 # Serialized .pkl ML models & SHAP plots (Generated)
├── scripts/              # Setup scripts for synthetic data & ML training
├── llm_eval.db           # SQLite Data Warehouse (Generated)
├── .env                  # API Keys for all 5+ LLM providers
└── requirements.txt      # Python dependencies
```

---

## 🧠 Deep Dive: Predictive ML Layer

Before an LLM request is sent to a provider, our secondary ML models intercept the payload:

* **Pre-Flight Failure Predictor (CatBoost):** Trained on historical benchmark data, it evaluates the `model_name`, `prompt_tokens`, `max_tokens`, and `temperature`. It returns the probability (0-100%) that the target model will completely fail to generate an exact match for the prompt.
* **Cost & Latency Forecasting (XGBoost):** Two highly optimized regression trees that dynamically estimate API costs and latency overhead *before* execution, allowing the system to fall back to a cheaper model if the forecasted cost exceeds a budget.
* **Automated Hallucination Detection (SentenceTransformers + XGBoost):** Generates dense embeddings of the Prompt + Ground Truth + Model Response. An XGBoost classifier evaluates the embedding vector to flag factual hallucinations without requiring expensive human or GPT-4 review.
* **Explainable AI (SHAP):** Every ML model is wrapped in a `shap.TreeExplainer` to extract feature importance (e.g. determining exactly how much `temperature=1.0` contributed to a hallucination flag).

---

## 📊 The Observability Dashboard

Access the interactive analytics control center (`http://localhost:8501`) to monitor your ecosystem:

* **Cost vs. Performance Pareto Frontier:** Scatter plot mapping all 99+ models. Instantly identify the most cost-efficient models for deployment (highest accuracy for the lowest cost).
* **Elo Leaderboards:** Head-to-head performance rankings calculated across all active benchmark categories (MMLU, GSM8k, HumanEval).
* **Experiment Comparison:** Side-by-side execution metrics calculating the delta across Exact Match %, Avg Latency, Cost per 1k, and Hallucination Rates between two specific models.
* **Error Clustering (K-Means):** Unsupervised clustering embeds questions that the models frequently fail on and groups them together to reveal systemic weaknesses (e.g., specific logic puzzles or coding paradigms).

---

## 🗄️ Database Schema (`llm_eval.db`)

All telemetry is normalized and stored locally via SQLAlchemy:

* **`models`**: Registry of available LLMs (e.g., `gpt-4o`, `claude-3-opus`).
* **`experiments` & `benchmark_runs`**: Groups specific configurations and datasets for tracked runs.
* **`questions`**: The ground-truth dataset containing the prompt and expected answer.
* **`responses`**: Tracks `latency_ms`, `prompt_tokens`, `completion_tokens`, and raw text.
* **`evaluation_results`**: Stores the output of the ML layer, including `exact_match`, `semantic_score`, and JSON `custom_metrics` (cost, hallucination_detected, quality_score).

---

## 📡 API Documentation (FastAPI Proxy)

**Base URL:** `http://localhost:8000`

**Endpoint:** `POST /v1/chat/completions`  
*Exactly mimics the OpenAI API format.*

**Request Body Example:**
```json
{
  "model": "anthropic/claude-3-opus-20240229",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in one sentence."}
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

The gateway intercepts this request, securely queries Anthropic via the Universal Connector, runs the predictive ML checks, logs the cost & latency to the database, and streams the standard response back to your client.

---

## 🚀 Setup & Run Instructions

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

Set up your API keys in the `.env` file at the root:
```env
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GEMINI_API_KEY=your_key
MISTRAL_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
```

### 2. Initialize the Database & Train Models
Populate the database with the model registry, generate massive synthetic telemetry, and train the local `CatBoost`/`XGBoost` models:
```bash
python3 scripts/seed_models.py
python3 scripts/generate_synthetic_data.py
python3 scripts/train_all_models.py
```
*(The `.pkl` model artifacts will be saved securely in `/data/`)*

### 3. Launch the Architecture

**Terminal 1 (The Gateway):**
```bash
python3 -m uvicorn app.main:app --reload --port 8000
```
**Terminal 2 (The Dashboard):**
```bash
python3 -m streamlit run app/visualization/dashboard.py
```

---

## 🔮 Future Roadmap (SaaS Commercialization)
- **Multi-Tenant PostgreSQL:** Migrate from local SQLite to a managed cloud database to support multiple isolated organizational workspaces.
- **Redis Semantic Caching:** Cache identical or semantically similar prompts at the edge to achieve 0ms latency and zero API cost.
- **Authentication & RBAC:** Implement JWT-based API key issuing and role-based access control for enterprise clients.
- **Next.js Frontend:** Transition the Streamlit analytics view into a fully productionized React/Next.js client interface.
