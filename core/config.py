class Config:
    """
    BASE CONFIG CLASS FOR LLMs.
    
    CAN BE EXTENDED TO ADD CUSTOM PARAMETERS.
    STORES PARAMETERS AS A DICTIONARY FOR FLEXIBILITY.
    """

    def __init__(self, **kwargs):
        """
        INITIALIZE CONFIG WITH ARBITRARY KEY-VALUE PARAMETERS.
        
        ALL PARAMETERS ARE STORED IN self.params DICTIONARY.
        """
        self.params = kwargs

    def get(self, key, default=None):
        """
        GET A CONFIG PARAMETER BY KEY, RETURN default IF NOT FOUND.
        """
        return self.params.get(key, default)

    def set(self, key, value):
        """
        SET OR UPDATE A CONFIG PARAMETER.
        """
        self.params[key] = value

    def to_dict(self):
        """
        RETURN ALL CONFIG PARAMETERS AS A DICTIONARY.
        """
        return dict(self.params)