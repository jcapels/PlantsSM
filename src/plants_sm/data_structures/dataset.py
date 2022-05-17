from abc import ABCMeta, abstractmethod
from typing import Any, List, Union

import numpy as np
import pandas as pd


class Dataset(metaclass=ABCMeta):

    def __init__(self, dataframe: Any,
                 representation_field: Union[str, List[str, int]] = None,
                 features_field: Union[str, List[str, int]] = None,
                 labels_field: Union[str, List[str, int]] = None,
                 instances_ids_field: Union[str, List[str, int]] = None):
        """
        Constructor

        Parameters
        ----------
        dataframe: Any
            dataframe to be consumed by the class and defined as class property
        representation_field: str | List[str | int] (optional)
            representation column field (to be processed)
        features_field: str | List[str | int] (optional)
            features column field
        labels_field: str | List[str | int] (optional)
            labels column field
        instances_ids_field: str | List[str | int] (optional)
            instances column field
        """
        self._instances = None
        self._representation_field = None
        self.representation_field = representation_field
        self._labels_names = None
        self.labels_field = labels_field
        self._features_names = None
        self.features_field = features_field
        self.instances_ids_field = instances_ids_field
        self._dataframe = None
        self.dataframe = dataframe

    @property
    def features_names(self) -> List[Any]:
        """
        Property for features names

        Returns
        -------
        list of the names of the features
        """
        return self._features_names

    @features_names.setter
    def features_names(self, value: List[Any]):
        """
        Setter for features names.

        Parameters
        ----------
        value : List[Any]
            list of the features names

        Returns
        -------

        """
        if isinstance(value, List):
            self._features_names = value

        else:
            raise TypeError("Feature names should be a list-like type.")

    @property
    @abstractmethod
    def features(self) -> np.array:
        """
        This property will be important for retrieving the columns of the dataframe with the features.
        
        Returns
        -------
        
        Features taken from the dataframe. ALIASES: X matrix/vector/array with the features.
        
        """
        pass

    @property
    def labels_names(self) -> List[Any]:
        """

        Returns
        -------

        """
        return self._labels_names

    @labels_names.setter
    def labels_names(self, value: List[Any]):
        """

        Parameters
        ----------
        value : list of the labels names to then be retrieved from the dataframe

        Returns
        -------

        """
        if isinstance(value, List):
            self._labels_names = value
        else:
            raise TypeError("Labels names should be a list-like type.")

    @property
    @abstractmethod
    def labels(self) -> np.array:
        """
        This property will contain the labels for supervised learning.

        Returns
        -------
        Labels for training and prediction. ALIASES: y vector with labels for classification and regression.
        """
        pass

    @property
    @abstractmethod
    def instances(self) -> np.array:
        """
        This property will contain the instances of the dataset.

        Returns
        -------
        Array with the instances.
        """
        pass

    @instances.setter
    def instances(self, value: List[Any]):
        """

        Parameters
        ----------
        value : list of the labels names to then be retrieved from the dataframe

        Returns
        -------

        """
        if isinstance(value, List):
            self._instances = value
        else:
            raise TypeError("Instances should be a list-like type.")

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

    @property
    def representation_field(self) -> Union[str, int]:
        """
        Property of the representation of the molecule, reaction or compounds

        Returns
        -------
        Representation field: str | int
            field where the biological entity is represented
        """
        return self._representation_field

    @representation_field.setter
    def representation_field(self, value: Union[str, int]):
        """
        Setter of the property. It verified the type of the value inputted.

        Parameters
        ----------
        value: str | int
            field where the biological entity is represented

        Returns
        -------

        """
        self._representation_field = value


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

    @property
    def features(self) -> np.array:
        """
        This property will only go to the dataframe and return a chunk with features.

        Returns
        -------
        features : array with the features

        """

        return np.array(self.dataframe.loc[:, self.features_names])

    @property
    def labels(self) -> np.array:
        return np.array(self.dataframe.loc[:, self.labels_names])
