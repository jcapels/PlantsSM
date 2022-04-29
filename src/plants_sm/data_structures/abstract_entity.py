from abc import ABCMeta, abstractmethod
from typing import Any


class AbstractEntity(metaclass=ABCMeta):
    """
    Abstract class to implement internal entities, e.g., proteins, dataframes, metabolites, reactions
    """

    @abstractmethod
    def convert(self, data: Any):
        """
        Abstract method that takes data of Any type and converts into the implemented class.

        Parameters
        ----------
        data: Any
            object of Any type that will be converted into the internal structure herein implemented (self)

        Returns
        -------

        """
        pass
