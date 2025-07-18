import asyncio
from .message import Message
from .config import Config
from llms.togetherllm import TogetherLLM
from llms.openrouterllm import OpenRouterLLM


class LLMManager:
    """
    MANAGES MULTIPLE LANGUAGE MODEL TYPES AND THEIR INSTANTIATED INSTANCES.
    
    - model_catalog: Dictionary mapping model type names to their classes (available models).
    - model_instances: Dictionary of instantiated models keyed by unique instance IDs.
    
    ALLOWS REGISTERING, USING, AND REMOVING MULTIPLE LLM INSTANCES SIMULTANEOUSLY.
    """

    def __init__(self):
        """
        INITIALIZES THE MANAGER WITH PREDEFINED AVAILABLE MODELS AND AN EMPTY INSTANCE STORE.
        """
        self.model_catalog: dict[str, type[LLM]] = {
            "TogetherLLM": TogetherLLM,
            "OpenRouterLLM": OpenRouterLLM,
        }
        self.model_instances: dict[str, LLM] = {}

    def instantiate_model(self, instance_id: str, model_type: str, config: Config):
        """
        CREATE A NEW LLM INSTANCE OF GIVEN TYPE WITH PROVIDED CONFIGURATION.
        
        PARAMETERS:
            instance_id (str): UNIQUE IDENTIFIER FOR THE NEW MODEL INSTANCE.
            model_type (str): TYPE NAME OF THE MODEL TO INSTANTIATE (MUST EXIST IN model_catalog).
            config (Config): CONFIGURATION OBJECT REQUIRED TO INITIALIZE THE MODEL.
        
        RAISES:
            ValueError: IF model_type IS NOT SUPPORTED OR instance_id ALREADY EXISTS.
        """
        if model_type not in self.model_catalog:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        if instance_id in self.model_instances:
            raise ValueError(f"Instance ID '{instance_id}' already exists.")
        model_class = self.model_catalog[model_type]
        self.model_instances[instance_id] = model_class(config)

    def remove_model(self, instance_id: str):
        """
        REMOVE AND CLEANUP THE LLM INSTANCE IDENTIFIED BY instance_id.
        
        PARAMETERS:
            instance_id (str): UNIQUE IDENTIFIER OF THE INSTANCE TO REMOVE.
        
        DOES NOTHING IF INSTANCE ID DOES NOT EXIST.
        """
        if instance_id in self.model_instances:
            self.model_instances[instance_id].reset_context()
            del self.model_instances[instance_id]

    async def use_model(self, instance_id: str, prompt: str, role: str = "user") -> str:
        """
        SEND A PROMPT TO THE SPECIFIED MODEL INSTANCE AND RETURN THE GENERATED RESPONSE.
        
        PARAMETERS:
            instance_id (str): UNIQUE IDENTIFIER OF THE MODEL INSTANCE TO USE.
            prompt (str): TEXT PROMPT TO SEND TO THE MODEL.
            role (str): ROLE OF THE PROMPT SENDER, DEFAULTS TO "user".
        
        RETURNS:
            str: THE MODEL GENERATED RESPONSE.
        
        RAISES:
            ValueError: IF instance_id IS NOT FOUND.
        """
        if instance_id not in self.model_instances:
            raise ValueError(f"Instance '{instance_id}' not found.")
        llm = self.model_instances[instance_id]
        llm.context.append(Message(role, prompt))
        response = await llm.generate(llm.context)
        llm.context.append(Message("assistant", response))
        return response

    async def use_multiple_models(self, instance_ids: list[str], prompt: str, role: str = "user") -> dict[str, str]:
        """
        SEND A PROMPT TO MULTIPLE MODEL INSTANCES IN PARALLEL AND RETURN THEIR RESPONSES.
        
        PARAMETERS:
            instance_ids (list[str]): LIST OF MODEL INSTANCE IDS TO QUERY.
            prompt (str): TEXT PROMPT TO SEND TO ALL MODELS.
            role (str): ROLE OF THE PROMPT SENDER, DEFAULTS TO "user".
        
        RETURNS:
            dict[str, str]: DICTIONARY MAPPING EACH INSTANCE ID TO ITS RESPONSE OR ERROR MESSAGE.
        """
        tasks = {
            instance_id: self.use_model(instance_id, prompt, role)
            for instance_id in instance_ids
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            instance_id: result if not isinstance(result, Exception) else f"Error: {result}"
            for instance_id, result in zip(tasks.keys(), results)
        }

    def get_model_catalog(self) -> list[str]:
        """
        RETURN A LIST OF ALL SUPPORTED MODEL TYPES AVAILABLE FOR INSTANTIATION.
        
        RETURNS:
            list[str]: MODEL TYPE NAMES.
        """
        return list(self.model_catalog.keys())

    def get_model_instances(self) -> list[str]:
        """
        RETURN A LIST OF ALL CURRENTLY INSTANTIATED MODEL INSTANCE IDS.
        
        RETURNS:
            list[str]: INSTANCE IDENTIFIERS.
        """
        return list(self.model_instances.keys())

    def print_conversation_history(self, instance_id: str):
        """
        PRINT THE CONVERSATION IN A READABLE DIALOGUE FORMAT.
        
        PARAMETERS:
            instance_id (str): UNIQUE IDENTIFIER OF THE MODEL INSTANCE.
        
        RAISES:
            ValueError: IF INSTANCE DOES NOT EXIST.
        """
        if instance_id not in self.model_instances:
            raise ValueError(f"Instance '{instance_id}' not found.")
        
        context = self.model_instances[instance_id].context
        print(f"--- CONVERSATION HISTORY FOR '{instance_id}' ---")
        for message in context:
            role = message.role.upper()
            print(f"{role}: {message.content}")
        print("--- END CONVERSATION ---")

    def print_all_conversation_histories(self):
        """
        DELEGATE TO print_conversation_history FOR ALL MODEL INSTANCES.
        """
        print("\n")
        for instance_id in self.get_model_instances():
            self.print_conversation_history(instance_id)
            print("\n")
