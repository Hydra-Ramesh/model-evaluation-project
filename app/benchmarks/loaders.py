import os
from typing import List
from datasets import load_dataset
from .base import BaseDatasetLoader, BenchmarkItem

class GSM8KLoader(BaseDatasetLoader):
    def download(self):
        # HuggingFace datasets library handles caching automatically
        load_dataset("gsm8k", "main", cache_dir=self.cache_dir)
        
    def load(self) -> List[BenchmarkItem]:
        dataset = load_dataset("gsm8k", "main", split="test", cache_dir=self.cache_dir)
        items = []
        for row in dataset:
            items.append(BenchmarkItem(
                benchmark_name="GSM8K",
                category="Math",
                question=row["question"],
                expected_answer=row["answer"],
                metadata={}
            ))
        return items

class MMLULoader(BaseDatasetLoader):
    def download(self):
        # MMLU has multiple subjects, we'll download a generic one or 'all' for this example
        load_dataset("cais/mmlu", "all", cache_dir=self.cache_dir)

    def load(self) -> List[BenchmarkItem]:
        dataset = load_dataset("cais/mmlu", "all", split="test", cache_dir=self.cache_dir)
        items = []
        for row in dataset:
            # MMLU typically provides a question, choices, and answer index.
            # We'll formulate a complete string for simplicity in the expected answer.
            choices = row["choices"]
            answer_idx = row["answer"]
            expected = choices[answer_idx] if isinstance(answer_idx, int) and answer_idx < len(choices) else str(answer_idx)
            
            items.append(BenchmarkItem(
                benchmark_name="MMLU",
                category=row["subject"],
                question=f"{row['question']}\nChoices: {', '.join(choices)}",
                expected_answer=expected,
                metadata={"choices": choices, "answer_index": answer_idx}
            ))
        return items

class HumanEvalLoader(BaseDatasetLoader):
    def download(self):
        load_dataset("openai_humaneval", cache_dir=self.cache_dir)

    def load(self) -> List[BenchmarkItem]:
        dataset = load_dataset("openai_humaneval", split="test", cache_dir=self.cache_dir)
        items = []
        for row in dataset:
            items.append(BenchmarkItem(
                benchmark_name="HumanEval",
                category="Coding",
                question=row["prompt"],
                expected_answer=row["canonical_solution"], # Or test code
                metadata={"task_id": row["task_id"], "test": row["test"], "entry_point": row["entry_point"]}
            ))
        return items

class TruthfulQALoader(BaseDatasetLoader):
    def download(self):
        load_dataset("truthful_qa", "generation", cache_dir=self.cache_dir)

    def load(self) -> List[BenchmarkItem]:
        dataset = load_dataset("truthful_qa", "generation", split="validation", cache_dir=self.cache_dir)
        items = []
        for row in dataset:
            items.append(BenchmarkItem(
                benchmark_name="TruthfulQA",
                category=row["category"],
                question=row["question"],
                expected_answer=row["best_answer"],
                metadata={"correct_answers": row["correct_answers"], "incorrect_answers": row["incorrect_answers"]}
            ))
        return items
