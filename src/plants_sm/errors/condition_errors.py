class PreConditionViolation(Exception):
    """
    Class for precondition violation exceptions.
    """
    def __init__(self, condition: str = None, message: str = "Precondition violation"):
        """
        Constructor for PreConditionViolation class.

        Parameters
        ----------
        condition: str
            condition that was violated
        message: str
            message to be displayed

        """
        self.condition = condition
        if condition:
            self.message = f"{message}: {condition}"
        else:
            self.message = message
        super().__init__(self.message)
