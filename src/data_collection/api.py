from abc import ABCMeta, abstractmethod
from typing import Any


class AbstractAPIAccessor(metaclass=ABCMeta):
    """
    Abstract API accessor, which aims at access to a specific API and retrieve information.
    """

    @abstractmethod
    def make_request(self, data: Any) -> int:
        """
        This abstract method aims at performing a request to the API in question, submitting data, such as identifiers.

        Parameters
        ----------
        data: Any
            data to be submitted to the API, e.g., protein identifiers or names

        Returns
        -------
        response_code: int
            HTTP response code of the request, it is an integer.
        """
        pass

    @abstractmethod
    @property
    def data(self):
        """
        Retrieved data from the API. It can be an object of the type Any.

        Returns
        -------

        """
        pass

    @abstractmethod
    def convert_data_to_entities(self):
        """
        Method to convert data to internal data structures

        Returns
        -------

        """
        pass
