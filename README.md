# 🤖 LLMManager

LLMManager is a flexible and extensible Python tool to register, manage, and interact with multiple **language model (LLM)** instances.

Supports:
- Dynamic model registration
- Config-driven instantiation
- Context-aware prompt/response handling
- Multi-model querying in parallel

---

## 📦 Features

- ✅ **Register** custom LLMs dynamically (no hardcoded types)
- 🚀 **Instantiate** models with unique configs
- 💬 **Send prompts** with optional memory (context)
- ⚡ **Query multiple models** in parallel
- 🧹 **Reset/remove** model instances on demand
- 📄 **Print** conversation history for any model

---

## 🧱 Structure

- `LLM`: Abstract base class for any language model.
- `LLMManager`: Manages LLM classes and their instances.
- `Config`: Holds model configuration (e.g., API keys, max tokens).
- `Message`: Represents a chat message with `role` and `content`.

---

## 🧑‍💻 Usage

### 1. Inherit from `LLM`

```python
from core.llm import LLM

class MyLLM(LLM):
    async def generate(self, messages: list[Message]) -> str:
        # Your generation logic here
        return "Hello from MyLLM!"
```
### 2. Register and Instantiate
```python
        manager = LLMManager()

        # Register your LLM class
        manager.register_model_type(MyLLM)

        # Create a config
        config = Config(api_key="sk-...", max_tokens=256)

        # Instantiate the model
        manager.instantiate_model("chatbot1", MyLLM, config)
```
### 3. Use the Model
```python
response = await manager.use_model("chatbot1", "What is the weather?")
print(response)
```
### 4. Use Multiple Models
```python
responses = await manager.use_multiple_models(
    ["chatbot1", "chatbot2"],
    prompt="Summarize this code",
    save_context=True,
    append_prompt=True
)
```
### 5. Conversation History
```python
manager.print_conversation_history("chatbot1")
manager.print_all_conversation_histories()
```

---

## 🧠 Requirements

- Python 3.10+
- Async-compatible LLM implementations

---

## ✅ Good to Know

- Models are keyed by their class, not strings → type-safe & clean
- You control what gets registered — no hardcoded logic
- Add memory/context handling via save_context and append_prompt

---

## 📬 Contributing

Feel free to open issues or submit pull requests! The goal is to support - plugin-based LLM development with minimal boilerplate.


