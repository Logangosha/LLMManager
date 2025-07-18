import asyncio
from typing import List
from core.message import Message
from core.config import Config
from core.llm import LLM
from core.llmmanager import LLMManager
import json

async def main():

    # INIT MANAGER
    manager = LLMManager()

    # GET API KEYS
    with open("secrets.json") as f:
        api_keys = json.load(f)

    # LLMS
    LLMS = {
        "togetherLLM": {
            "id": "togetherLLM_1",
            "class_name": "TogetherLLM",
            "config": Config(
                api_key=api_keys["together_api_key"],
                model="mistralai/Mixtral-8x7B-Instruct-v0.1"
            )
        },
        "openRouterLLM": {
            "id": "openRouterLLM_1",
            "class_name": "OpenRouterLLM",
            "config": Config(
                api_key=api_keys["openrouter_api_key"],
                model="deepseek/deepseek-r1-0528:free"
            )
        }
    }

    # INIT MODELS
    for key, params in LLMS.items():
        manager.instantiate_model(params["id"], params["class_name"], params["config"])

    # PROMPT MODELS
    prompt = "Please tell me what you are called."
    await manager.use_multiple_models([params["id"] for params in LLMS.values()], prompt) 
    await manager.use_model(LLMS["openRouterLLM"]["id"], "What?")

    # PRINT RESPONCES
    manager.print_all_conversation_histories()

    for llm in LLMS:
        manager.remove_model(llm)

if __name__ == "__main__":
    asyncio.run(main())