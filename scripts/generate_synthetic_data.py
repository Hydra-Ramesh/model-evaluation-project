import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import SessionLocal, engine
from app.database.models import Model, Experiment, BenchmarkRun, EvaluationResult, Question, Response, Base

def generate_synthetic_data(num_records=2000):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    models = db.query(Model).all()
    if not models:
        print("Please run seed_models.py first.")
        return
        
    exp = Experiment(name="Massive Synthetic Run", description="Mock data for ML training", config={})
    db.add(exp)
    db.commit()
    db.refresh(exp)
    
    benchmarks = ["gsm8k", "mmlu", "humaneval", "truthfulqa", "ifeval", "halueval"]
    categories = ["Arithmetic Errors", "Logical Errors", "Hallucinations", "Coding Bugs", "Instruction Following Failures"]
    
    runs = []
    for model in models:
        for b in benchmarks:
            run = BenchmarkRun(experiment_id=exp.id, model_id=model.id, benchmark_name=b)
            db.add(run)
            runs.append((run, model, b))
    db.commit()
    
    print(f"Generating {num_records} evaluation results...")
    
    for i in range(num_records):
        run, model, bench = random.choice(runs)
        
        q = Question(
            benchmark_name=bench,
            category=random.choice(categories),
            question_text=f"Synthetic question {i} for {bench} testing.",
            expected_answer=f"Expected factual answer {i}."
        )
        db.add(q)
        db.commit()
        db.refresh(q)
        
        prompt_len = random.randint(50, 1500)
        max_tokens = random.choice([512, 1024, 2048, 4096])
        temperature = random.choice([0.0, 0.3, 0.7, 1.0])
        
        is_high_tier = any(x in model.name.lower() for x in ["gpt-4", "opus", "pro", "max", "large", "70b", "405b", "v3", "o3"])
        acc_prob = 0.85 if is_high_tier else 0.55
        accuracy = 1.0 if random.random() < acc_prob else 0.0
        hallucinated = True if accuracy == 0.0 and random.random() < 0.6 else False
        
        latency = random.uniform(0.5, 4.0) + (prompt_len / 300.0)
        if not is_high_tier:
            latency -= random.uniform(0.2, 0.8)
            
        cost = prompt_len * random.uniform(0.000001, 0.00001) + max_tokens * random.uniform(0.000002, 0.00002)
        if not is_high_tier: cost *= 0.1
        
        res = Response(
            run_id=run.id,
            question_id=q.id,
            response_text=f"The model responded with output {i} which is {'correct' if accuracy else 'incorrect'}.",
            latency_ms=max(100.0, latency * 1000),
            prompt_tokens=prompt_len,
            completion_tokens=random.randint(10, max_tokens)
        )
        db.add(res)
        db.commit()
        db.refresh(res)
        
        quality_score = random.uniform(8.0, 10.0) if accuracy == 1.0 else random.uniform(1.0, 5.0)
        ev = EvaluationResult(
            response_id=res.id,
            exact_match=bool(accuracy),
            semantic_score=accuracy if accuracy > 0 else random.uniform(0.1, 0.5),
            custom_metrics={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "quality_score": quality_score,
                "hallucination_detected": hallucinated,
                "cost": cost
            }
        )
        db.add(ev)
        
        if i % 500 == 0:
            print(f"Inserted {i} records...")
            db.commit()
            
    db.commit()
    print("Generation complete!")

if __name__ == "__main__":
    generate_synthetic_data()
