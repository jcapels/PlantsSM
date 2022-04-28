from abc import ABCMeta, abstractmethod
from typing import Any

import pandas as pd


class Dataset(metaclass=ABCMeta):

    def __init__(self, dataframe: Any):
        """
        
        Parameters
        ----------
        dataframe: Any
            dataframe to be consumed by the class and defined as class property
        """
        self._dataframe = None
        self.dataframe = dataframe

    @property
    def dataframe(self) -> Any:
        """
        Property of all datasets: they should have an associated dataframe.

        Returns
        -------
        dataframe : Any
            dataframe with the required data
        """
        return self._dataframe

    @abstractmethod
    def _set_dataframe(self, value: Any):
        """
        Just a private method to verify the true type of the dataframe according to the type of dataset.

        Parameters
        ----------
        value: Any
            dataframe to be set, it can be in pd.DataFrame format, but can also be a List or Dictionary 
            (it can be specific for each data type)

        Returns
        -------

        """
        pass

    @dataframe.setter
    def dataframe(self, value: Any):
        """
        Setter of the property. It verified the type of the value inputted.

        Parameters
        ----------
        value: Any
            dataframe to be set, it can be in pd.DataFrame format, but can also be a List or Dictionary 
            (it can be specific for each data type)

        Returns
        -------

        """
        self._set_dataframe(value)


class PandasDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame):
        super().__init__(dataframe)

    def _set_dataframe(self, value: pd.DataFrame):
        """
        Setter of the property. It verified the type of the value inputted.

        Parameters
        ----------
        value: Any
            dataframe to be set, it can be in pd.DataFrame format, but can also be a List or Dictionary 
            (it can be specific for each data type)

        Returns
        -------

        """
        if isinstance(value, pd.DataFrame):
            self._dataframe = value
        else:
            raise TypeError("It seems that the type of your input is not a pandas DataFrame."
                            "The type of the dataframe should be a pandas DataFrame")
