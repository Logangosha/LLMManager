class Message:
    """
    REPRESENTS A SINGLE MESSAGE IN A CHAT CONTEXT.
    
    INCLUDES WHO SENT THE MESSAGE (ROLE) AND THE TEXT CONTENT.
    """
    def __init__(self, role: str, content: str):
        """
        INITIALIZE A MESSAGE OBJECT.
        
        :param role: WHO SENT THE MESSAGE (E.G., 'user', 'assistant', 'system').
        :param content: THE TEXT CONTENT OF THE MESSAGE.
        """
        self.role = role
        self.content = content