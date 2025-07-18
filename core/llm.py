from abc import ABC, abstractmethod
from typing import List
from .message import Message
from .config import Config

class LLM(ABC):
    """
    ABSTRACT BASE CLASS FOR ALL LANGUAGE MODELS.
    
    FUTURE DEVELOPERS SHOULD EXTEND THIS CLASS AND IMPLEMENT THE ABSTRACT METHODS.
    """

    def __init__(self, config: Config):
        """
        INITIALIZE THE LLM WITH A CONFIG OBJECT.
        
        :param config: Config instance holding model-specific parameters.
        """
        self.config = config
        self.context: List[Message] = []

    @abstractmethod
    async def generate(self, messages: List[Message]) -> str:
        """
        ABSTRACT METHOD TO GENERATE A RESPONSE GIVEN A LIST OF MESSAGES.
        
        MUST BE OVERRIDDEN BY SUBCLASSES.
        
        :param messages: List of Message objects representing the conversation context.
        :return: Generated response string.
        """
        pass

    def reset_context(self):
        """
        CLEAR THE CONTEXT MESSAGES.
        """
        self.context.clear()

    def update_config(self, **kwargs):
        """
        UPDATE CONFIG PARAMETERS DYNAMICALLY.
        
        :param kwargs: Key-value pairs to update config params.
        """
        for key, value in kwargs.items():
            self.config.set(key, value)
