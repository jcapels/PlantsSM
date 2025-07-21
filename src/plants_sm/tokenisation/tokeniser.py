from abc import abstractmethod
from typing import List, Union

from pydantic import BaseModel


class Tokenizer(BaseModel):
    """
    Tokenizer class that is used to encode and decode sequences.
    """

    class Config:
        """
        Model Configuration: https://pydantic-docs.helpmanual.io/usage/model_config/
        """
        extra = 'allow'
        allow_mutation = True
        validate_assignment = True
        underscore_attrs_are_private = True

    @abstractmethod
    def tokenize(self, sequence: str) -> List[str]:
        """
        Abstract method that has to be implemented by all tokenizers to tokenize a sequence.
        """


