from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.database.connection import engine, Base

# Ensure all database tables are created synchronously on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="LLM Evaluation & Benchmarking Platform API",
    description="A centralized orchestrator for evaluating LLM reasoning, coding, factuality, and costs.",
    version="1.0.0"
)

# Register endpoints
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the LLM Evaluation Platform. View /docs for the interactive API reference."
    }
