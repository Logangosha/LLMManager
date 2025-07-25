import asyncio
from .message import Message
from .config import Config
from .llm import LLM


class LLMManager:
    """
    MANAGES MULTIPLE LANGUAGE MODEL TYPES AND THEIR INSTANTIATED INSTANCES.

    - MODEL_CATALOG: DICTIONARY MAPPING MODEL CLASSES TO THEMSELVES (AVAILABLE MODELS).
    - MODEL_INSTANCES: DICTIONARY OF INSTANTIATED MODELS KEYED BY UNIQUE INSTANCE IDS.

    ALLOWS REGISTERING, USING, AND REMOVING MULTIPLE LLM INSTANCES SIMULTANEOUSLY.
    """

    def __init__(self):
        """
        INITIALIZES THE MANAGER WITH EMPTY MODEL CATALOG AND INSTANCE STORE.
        """
        self.model_catalog: dict[type[LLM], type[LLM]] = {}
        self.model_instances: dict[str, LLM] = {}

    def register_model_type(self, model_class: type[LLM]):
        """
        REGISTER A NEW MODEL TYPE.

        :PARAM model_class: CLASS REFERENCE (MUST INHERIT FROM LLM).
        :RAISES ValueError: IF model_class IS NOT A SUBCLASS OF LLM OR ALREADY REGISTERED.
        """
        if not issubclass(model_class, LLM):
            raise ValueError("MODEL_CLASS MUST INHERIT FROM LLM.")
        if model_class in self.model_catalog:
            raise ValueError(f"MODEL CLASS '{model_class.__name__}' ALREADY REGISTERED.")
        self.model_catalog[model_class] = model_class

    def instantiate_model(self, instance_id: str, model_type: type[LLM], config: Config):
        """
        CREATE A NEW LLM INSTANCE OF GIVEN TYPE WITH PROVIDED CONFIGURATION.

        :PARAM instance_id: UNIQUE IDENTIFIER FOR THE NEW MODEL INSTANCE.
        :PARAM model_type: LLM CLASS TO INSTANTIATE (MUST BE REGISTERED).
        :PARAM config: CONFIG OBJECT REQUIRED TO INITIALIZE THE MODEL.

        :RAISES ValueError: IF model_type IS NOT REGISTERED OR instance_id ALREADY EXISTS.
        """
        if model_type not in self.model_catalog:
            raise ValueError(f"MODEL CLASS '{model_type.__name__}' IS NOT REGISTERED.")
        if instance_id in self.model_instances:
            raise ValueError(f"INSTANCE ID '{instance_id}' ALREADY EXISTS.")
        self.model_instances[instance_id] = model_type(config)

    def remove_model(self, instance_id: str):
        """
        REMOVE AND CLEAN UP THE LLM INSTANCE IDENTIFIED BY instance_id.

        :PARAM instance_id: UNIQUE IDENTIFIER OF THE INSTANCE TO REMOVE.

        DOES NOTHING IF INSTANCE ID DOES NOT EXIST.
        """
        if instance_id in self.model_instances:
            self.model_instances[instance_id].reset_context()
            del self.model_instances[instance_id]

    async def use_model(
        self,
        instance_id: str,
        prompt: str,
        role: str = "user",
        save_context: bool = False,
        append_prompt: bool = False
    ) -> str:
        """
        SEND A PROMPT TO THE SPECIFIED MODEL INSTANCE AND RETURN THE GENERATED RESPONSE.

        :PARAM instance_id: UNIQUE IDENTIFIER OF THE MODEL INSTANCE TO USE.
        :PARAM prompt: TEXT PROMPT TO SEND TO THE MODEL.
        :PARAM role: ROLE OF THE PROMPT SENDER (DEFAULT "user").
        :PARAM save_context: IF TRUE, APPENDS PROMPT AND RESPONSE TO CONTEXT.
        :PARAM append_prompt: IF TRUE, APPENDS PROMPT TO CONTEXT BEFORE GENERATION.

        :RETURNS: GENERATED RESPONSE FROM THE MODEL.

        :RAISES ValueError: IF instance_id IS NOT FOUND.
        """
        if instance_id not in self.model_instances:
            raise ValueError(f"INSTANCE '{instance_id}' NOT FOUND.")
        llm = self.model_instances[instance_id]

        context_to_use = llm.context.copy()
        if append_prompt:
            context_to_use.append(Message(role, prompt))

        response = await llm.generate(context_to_use)

        if save_context:
            if append_prompt:
                llm.context.append(Message(role, prompt))
            llm.context.append(Message("assistant", response))

        return response

    async def use_multiple_models(
        self,
        instance_ids: list[str],
        prompt: str,
        role: str = "user",
        save_context: bool = False,
        append_prompt: bool = False,
    ) -> dict[str, str]:
        """
        SEND A PROMPT TO MULTIPLE MODEL INSTANCES IN PARALLEL AND RETURN THEIR RESPONSES.

        :PARAM instance_ids: LIST OF INSTANCE IDS TO QUERY.
        :PARAM prompt: TEXT PROMPT TO SEND.
        :PARAM role: PROMPT SENDER ROLE (DEFAULT "user").
        :PARAM save_context: IF TRUE, APPEND PROMPT/RESPONSE TO CONTEXT.
        :PARAM append_prompt: IF TRUE, ADD PROMPT TO CONTEXT BEFORE GENERATION.

        :RETURNS: DICTIONARY MAPPING INSTANCE IDS TO RESPONSES OR ERROR MESSAGES.
        """
        tasks = {
            instance_id: self.use_model(instance_id, prompt, role, save_context, append_prompt)
            for instance_id in instance_ids
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            instance_id: result if not isinstance(result, Exception) else f"ERROR: {result}"
            for instance_id, result in zip(tasks.keys(), results)
        }

    def get_model_catalog(self) -> list[str]:
        """
        RETURN A LIST OF ALL SUPPORTED MODEL CLASSES AVAILABLE FOR INSTANTIATION.

        :RETURNS: LIST OF MODEL CLASS NAMES.
        """
        return [cls.__name__ for cls in self.model_catalog.keys()]

    def get_model_instances(self) -> list[str]:
        """
        RETURN A LIST OF ALL CURRENTLY INSTANTIATED MODEL INSTANCE IDS.

        :RETURNS: LIST OF INSTANCE IDENTIFIERS.
        """
        return list(self.model_instances.keys())

    def print_conversation_history(self, instance_id: str):
        """
        PRINT THE CONVERSATION IN A READABLE DIALOGUE FORMAT.

        :PARAM instance_id: UNIQUE IDENTIFIER OF THE MODEL INSTANCE.

        :RAISES ValueError: IF INSTANCE DOES NOT EXIST.
        """
        if instance_id not in self.model_instances:
            raise ValueError(f"INSTANCE '{instance_id}' NOT FOUND.")

        context = self.model_instances[instance_id].context
        print(f"--- CONVERSATION HISTORY FOR '{instance_id}' ---")
        for message in context:
            role = message.role.upper()
            print(f"{role}: {message.content}")
        print("--- END CONVERSATION ---")

    def print_all_conversation_histories(self):
        """
        PRINT ALL CONVERSATION HISTORIES FOR EVERY MODEL INSTANCE.
        """
        print("\n")
        for instance_id in self.get_model_instances():
            self.print_conversation_history(instance_id)
            print("\n")
