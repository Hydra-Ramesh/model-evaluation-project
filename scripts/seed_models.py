import os
import sys

# Add parent dir to path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import SessionLocal, engine
from app.database.models import Model, Base

def seed_models():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    models_to_seed = [
        # OpenAI
        {"name": "gpt-5", "provider": "openai", "parameters": "Unknown"},
        {"name": "gpt-5.5", "provider": "openai", "parameters": "Unknown"},
        {"name": "gpt-5-mini", "provider": "openai", "parameters": "Unknown"},
        {"name": "gpt-5-nano", "provider": "openai", "parameters": "Unknown"},
        {"name": "gpt-4.1", "provider": "openai", "parameters": "Unknown"},
        {"name": "gpt-4.1-mini", "provider": "openai", "parameters": "Unknown"},
        {"name": "gpt-4.1-nano", "provider": "openai", "parameters": "Unknown"},
        {"name": "gpt-4o", "provider": "openai", "parameters": "Unknown"},
        {"name": "gpt-4o-mini", "provider": "openai", "parameters": "Unknown"},
        {"name": "o3", "provider": "openai", "parameters": "Reasoning"},
        {"name": "o3-pro", "provider": "openai", "parameters": "Reasoning"},
        {"name": "o4-mini", "provider": "openai", "parameters": "Reasoning"},
        {"name": "codex", "provider": "openai", "parameters": "Code"},
        
        # Anthropic
        {"name": "claude-3-opus-20240229", "provider": "anthropic", "parameters": "Unknown"},
        {"name": "claude-3-sonnet-20240229", "provider": "anthropic", "parameters": "Unknown"},
        {"name": "claude-3-haiku-20240307", "provider": "anthropic", "parameters": "Unknown"},
        {"name": "claude-fable-5", "provider": "anthropic", "parameters": "Unknown"},
        {"name": "claude-mythos-5", "provider": "anthropic", "parameters": "Unknown"},
        
        # Google DeepMind
        {"name": "gemini/gemini-2.0-flash", "provider": "google", "parameters": "Unknown"},
        {"name": "gemini/gemini-2.5-flash", "provider": "google", "parameters": "Unknown"},
        {"name": "gemini/gemini-2.5-pro", "provider": "google", "parameters": "Unknown"},
        {"name": "gemini/gemini-3-pro", "provider": "google", "parameters": "Unknown"},
        {"name": "gemini/gemini-3.1-pro", "provider": "google", "parameters": "Unknown"},
        {"name": "gemini/gemma-3", "provider": "google", "parameters": "Unknown"},
        {"name": "gemini/gemma-4", "provider": "google", "parameters": "Unknown"},
        
        # Meta
        {"name": "together_ai/meta-llama/Llama-3-70b-chat-hf", "provider": "meta", "parameters": "70B"},
        {"name": "together_ai/meta-llama/Llama-3.1-405b-instruct", "provider": "meta", "parameters": "405B"},
        {"name": "together_ai/meta-llama/Llama-4-scout", "provider": "meta", "parameters": "Unknown"},
        {"name": "together_ai/meta-llama/Llama-4-maverick", "provider": "meta", "parameters": "Unknown"},
        {"name": "together_ai/meta-llama/Llama-4-behemoth", "provider": "meta", "parameters": "Unknown"},
        {"name": "together_ai/meta-llama/muse-spark", "provider": "meta", "parameters": "Unknown"},
        
        # xAI
        {"name": "grok-3", "provider": "xai", "parameters": "Unknown"},
        {"name": "grok-4", "provider": "xai", "parameters": "Unknown"},
        {"name": "grok-reasoning-variants", "provider": "xai", "parameters": "Reasoning"},
        
        # DeepSeek
        {"name": "deepseek/deepseek-chat", "provider": "deepseek", "parameters": "Unknown"},
        {"name": "deepseek/deepseek-reasoner", "provider": "deepseek", "parameters": "Unknown"},
        {"name": "deepseek/deepseek-r1", "provider": "deepseek", "parameters": "Unknown"},
        {"name": "deepseek/deepseek-v3", "provider": "deepseek", "parameters": "Unknown"},
        {"name": "deepseek/deepseek-v3.1", "provider": "deepseek", "parameters": "Unknown"},
        {"name": "deepseek/deepseek-v4-flash", "provider": "deepseek", "parameters": "Unknown"},
        {"name": "deepseek/deepseek-v4-pro", "provider": "deepseek", "parameters": "Unknown"},

        # Alibaba / Qwen
        {"name": "together_ai/Qwen/Qwen2.5-72B-Instruct", "provider": "alibaba", "parameters": "72B"},
        {"name": "together_ai/Qwen/Qwen3", "provider": "alibaba", "parameters": "Unknown"},
        {"name": "together_ai/Qwen/Qwen3-Max", "provider": "alibaba", "parameters": "Unknown"},
        {"name": "together_ai/Qwen/Qwen3-Max-Thinking", "provider": "alibaba", "parameters": "Reasoning"},
        {"name": "together_ai/Qwen/Qwen-Coder", "provider": "alibaba", "parameters": "Code"},
        {"name": "together_ai/Qwen/Qwen-VL", "provider": "alibaba", "parameters": "Vision"},
        {"name": "together_ai/Qwen/Qwen-Math", "provider": "alibaba", "parameters": "Math"},

        # Zhipu AI
        {"name": "zhipu/glm-4", "provider": "zhipu", "parameters": "Unknown"},
        {"name": "zhipu/glm-4.7", "provider": "zhipu", "parameters": "Unknown"},
        {"name": "zhipu/glm-5", "provider": "zhipu", "parameters": "Unknown"},
        {"name": "zhipu/glm-5.1", "provider": "zhipu", "parameters": "Unknown"},

        # Mistral
        {"name": "mistral/mistral-large-latest", "provider": "mistral", "parameters": "Unknown"},
        {"name": "mistral/mistral-medium", "provider": "mistral", "parameters": "Unknown"},
        {"name": "mistral/mistral-small", "provider": "mistral", "parameters": "Unknown"},
        {"name": "mistral/mistral-small-4", "provider": "mistral", "parameters": "Unknown"},
        {"name": "mistral/mixtral-8x7b-instruct", "provider": "mistral", "parameters": "8x7B"},
        {"name": "mistral/codestral-latest", "provider": "mistral", "parameters": "Code"},
        {"name": "mistral/ministral", "provider": "mistral", "parameters": "Unknown"},

        # Microsoft
        {"name": "azure/phi-3", "provider": "microsoft", "parameters": "Unknown"},
        {"name": "azure/phi-4", "provider": "microsoft", "parameters": "Unknown"},
        {"name": "azure/phi-4-mini", "provider": "microsoft", "parameters": "Unknown"},
        {"name": "azure/phi-4-multimodal", "provider": "microsoft", "parameters": "Vision"},

        # Cohere
        {"name": "cohere/command-r", "provider": "cohere", "parameters": "Unknown"},
        {"name": "cohere/command-r-plus", "provider": "cohere", "parameters": "Unknown"},
        {"name": "cohere/command-a", "provider": "cohere", "parameters": "Unknown"},

        # Amazon
        {"name": "bedrock/amazon.nova-lite", "provider": "amazon", "parameters": "Unknown"},
        {"name": "bedrock/amazon.nova-pro", "provider": "amazon", "parameters": "Unknown"},
        {"name": "bedrock/amazon.nova-premier", "provider": "amazon", "parameters": "Unknown"},

        # IBM
        {"name": "watsonx/ibm/granite-3", "provider": "ibm", "parameters": "Unknown"},
        {"name": "watsonx/ibm/granite-code", "provider": "ibm", "parameters": "Code"},
        {"name": "watsonx/ibm/granite-vision", "provider": "ibm", "parameters": "Vision"},

        # NVIDIA
        {"name": "nvidia/nemotron-4-340b", "provider": "nvidia", "parameters": "340B"},
        {"name": "nvidia/llama-nemotron-family", "provider": "nvidia", "parameters": "Unknown"},

        # Databricks
        {"name": "databricks/dbrx-instruct", "provider": "databricks", "parameters": "132B"},

        # Allen AI
        {"name": "together_ai/allenai/OLMo", "provider": "allenai", "parameters": "Unknown"},
        {"name": "together_ai/allenai/OLMo-2", "provider": "allenai", "parameters": "Unknown"},

        # TII
        {"name": "together_ai/tiiuae/falcon-7b", "provider": "tii", "parameters": "7B"},
        {"name": "together_ai/tiiuae/falcon-40b", "provider": "tii", "parameters": "40B"},
        {"name": "together_ai/tiiuae/falcon-180b", "provider": "tii", "parameters": "180B"},

        # Moonshot AI
        {"name": "moonshot/kimi", "provider": "moonshot", "parameters": "Unknown"},
        {"name": "moonshot/kimi-k2", "provider": "moonshot", "parameters": "Unknown"},
        {"name": "moonshot/kimi-k2.5", "provider": "moonshot", "parameters": "Unknown"},

        # MiniMax
        {"name": "minimax/minimax-text", "provider": "minimax", "parameters": "Unknown"},
        {"name": "minimax/minimax-m1", "provider": "minimax", "parameters": "Unknown"},

        # Baichuan
        {"name": "baichuan/baichuan-2", "provider": "baichuan", "parameters": "Unknown"},
        {"name": "baichuan/baichuan-4", "provider": "baichuan", "parameters": "Unknown"},

        # 01.AI
        {"name": "together_ai/01-ai/Yi-34B-Chat", "provider": "01ai", "parameters": "34B"},
        {"name": "together_ai/01-ai/Yi-Large", "provider": "01ai", "parameters": "Unknown"},

        # AI21
        {"name": "ai21/jamba", "provider": "ai21", "parameters": "Unknown"},
        {"name": "ai21/jamba-large", "provider": "ai21", "parameters": "Unknown"},

        # Stability AI
        {"name": "together_ai/stabilityai/stablelm-2", "provider": "stabilityai", "parameters": "Unknown"},

        # Groq (Ultra-fast Inference)
        {"name": "groq/llama3-8b-8192", "provider": "groq", "parameters": "8B"},
        {"name": "groq/llama3-70b-8192", "provider": "groq", "parameters": "70B"},
        {"name": "groq/mixtral-8x7b-32768", "provider": "groq", "parameters": "8x7B"},
        {"name": "groq/gemma-7b-it", "provider": "groq", "parameters": "7B"},

        # India Models
        {"name": "bharatgen/param-2", "provider": "bharatgen", "parameters": "Unknown"},
        {"name": "together_ai/sarvamai/sarvam-30b", "provider": "sarvamai", "parameters": "30B"},
        {"name": "together_ai/sarvamai/sarvam-105b", "provider": "sarvamai", "parameters": "105B"},
    ]
    
    added_count = 0
    for model_data in models_to_seed:
        # Check if exists
        existing = db.query(Model).filter(Model.name == model_data["name"]).first()
        if not existing:
            new_model = Model(**model_data)
            db.add(new_model)
            added_count += 1
            
    db.commit()
    db.close()
    print(f"Successfully seeded {added_count} models.")

if __name__ == "__main__":
    seed_models()
