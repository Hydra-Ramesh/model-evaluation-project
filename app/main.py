from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.database.connection import engine, Base

import sys
import traceback

# Ensure all database tables are created synchronously on startup
try:
    print("Attempting to connect to database and create tables...", flush=True)
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!", flush=True)
except Exception as e:
    print(f"CRITICAL ERROR DURING DATABASE INITIALIZATION: {str(e)}", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()

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
