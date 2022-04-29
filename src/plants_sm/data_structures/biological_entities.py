from abc import abstractmethod
from typing import Any, Union, Dict, List

from plants_sm.data_structures.abstract_entity import AbstractEntity


class BiologicalEntity(AbstractEntity):
    """
    Class that represents all biological entities.
    Biological entities should be instantiated under very specific conditions, e.g., to visualise structures or check
    information on this entity.
    """

    def __init__(self):
        self._representation = None

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

    @property
    def representation(self):
        """
        Representation of the biological entity. It can be a dictionary of multiple types of representations,
        a string, an array.
        Returns
        -------

        """
        return self._representation

    @representation.setter
    def representation(self, value: Union[Dict, List, str]):
        """

        Parameters
        ----------
        value : Dict | List | str
            value to be set to the representation property

        Returns
        -------

        """
        if isinstance(value, Dict) or isinstance(value, List) or isinstance(value, str):
            self._representation = value
        else:
            raise TypeError("You are not setting a correct type to your representation. "
                            "It should be a dictionary a list or a string")


class Protein(BiologicalEntity):
    """
    Class to represent a protein.
    """

    def __init__(self):
        super().__init__()

    def convert(self, data: Any):
        """
        Method to convert any type of data into a Protein internal class.
        It is mandatory that the representation property is herein implemented.
        Other properties can also be defined here, such as external references to databases and literature, name,
        properties, etc.

        Parameters
        ----------
        data

        Returns
        -------

        """
        raise NotImplementedError


class Compound(BiologicalEntity):
    """
    Class to represent a protein.
    """

    def __init__(self):
        super().__init__()

    def convert(self, data: Any):
        """
        Method to convert any type of data into a Compound internal class.
        It is mandatory that the representation property is herein implemented.
        Other properties can also be defined here, such as external references to databases and literature, name,
        properties, etc.

        Parameters
        ----------
        data

        Returns
        -------

        """
        raise NotImplementedError


class Reaction(BiologicalEntity):
    """
    Class to represent a protein.
    """

    def __init__(self):
        super().__init__()

    def convert(self, data: Any):
        """
        Method to convert any type of data into a Reaction internal class.
        It is mandatory that the representation property is herein implemented.
        Other properties can also be defined here, such as external references to databases and literature, name,
        properties, etc.

        Parameters
        ----------
        data

        Returns
        -------

        """
        raise NotImplementedError
