from abc import ABCMeta, abstractmethod
from typing import Any, List, Union

import numpy as np
import pandas as pd

from plants_sm.mixins.mixins import CSVMixin, ExcelMixin


class Dataset(metaclass=ABCMeta):

    def __init__(self, dataframe: Any,
                 representation_field: Union[str, List[Union[str, int]]] = None,
                 features_field: Union[str, List[Union[str, int]]] = None,
                 labels_field: Union[str, List[Union[str, int]]] = None,
                 instances_ids_field: Union[str, List[Union[str, int]]] = None):
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

        # the features fields is a list of fields that are used to extract the features
        # however, they can be both strings or integers, so we need to check the type of the field
        # and convert it to a list of strings
        self._features_names = None
        if not isinstance(features_field, List):
            self.features_field = [features_field]
        else:
            self.features_field = features_field

        # the instance ids field is defined here, however, if it is None, eventually is derived from the dataframe
        # setter
        self.instances_ids_field = instances_ids_field

        # the dataframe setter will derive the instance ids field if it is None
        self._dataframe = None
        self.dataframe = dataframe

        self._representation_field = None
        self.representation_field = representation_field

        self._labels_names = None
        if not isinstance(labels_field, List):
            self.labels_names = [labels_field]
        else:
            self.labels_names = labels_field

        if self.features_field is not None:
            self._set_features_names()

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
    def identifiers(self) -> List[Union[str, int]]:
        """
        Property for identifiers. It should return the identifiers of the dataset.

        Returns
        -------
        list of the identifiers: List[Union[str, int]]
        """
        pass

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

    def set_dataframe_dependent_methods(self, func):
        def wrap(*args, **kwargs):
            func(*args, **kwargs)

            return self

        return wrap

    def set_dataframe(self, value: Any):
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
        if value is not None:
            self._set_dataframe(value)
            self._set_features_names()
            if self.instances_ids_field is None:
                self._set_instances_ids_field()

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
        self.set_dataframe(value)

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

    @abstractmethod
    def _set_instances_ids_field(self):
        """
        Private method to set the instances ids field if it is not defined.

        """
        pass

    @abstractmethod
    def _set_features_names(self):
        """
        Private method to set the features names if it is not defined.

        """
        pass

    @abstractmethod
    def _set_dataframe(self, value: Any):
        """
        Private method to set the dataframe.

        Parameters
        ----------
        value: Any
            dataframe to be set, it can be in pd.DataFrame format, but can also be a List or Dictionary
            (it can be specific for each data type)

        """
        pass


class PandasDataset(Dataset, CSVMixin, ExcelMixin):

    def __init__(self, dataframe: pd.DataFrame = None,
                 representation_field: Union[str, List[Union[str, int]]] = None,
                 features_field: Union[str, List[Union[str, int]]] = None,
                 labels_field: Union[str, List[Union[str, int]]] = None,
                 instances_ids_field: Union[str, List[Union[str, int]]] = None
                 ):

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

        super().__init__(dataframe, representation_field, features_field, labels_field, instances_ids_field)

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
        if isinstance(value, pd.DataFrame) or value is None:
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
        if self.features_names is not None:
            return np.array(self.dataframe.loc[:, self.features_names])
        else:
            raise ValueError("The features were not extracted yet.")

    @property
    def labels(self) -> np.array:
        return np.array(self.dataframe.loc[:, self.labels_names])

    @property
    def instances(self) -> np.array:
        return np.array(self.dataframe.loc[:, self.representation_field])

    @property
    def identifiers(self) -> List[str]:
        return self.dataframe.loc[:, self.instances_ids_field]

    def _set_features_names(self):
        """
        Private method to set the features names if it is not defined.
        """
        if self.features_names is None:
            if isinstance(self.features_field[0], str):
                self.features_names = list(self.features_field)
            else:
                self.features_names = []
                for feature in self.features_field:
                    self.features_names.append(self.dataframe.columns[feature])

    def _set_instances_ids_field(self):
        """
        Private method to set the instances ids field if it is not defined.

        """
        if self.instances_ids_field is None:
            self.instances_ids_field = "identifier"
            self.dataframe["identifier"] = list(range(self.dataframe.shape[0]))
            self.dataframe["identifier"] = self.dataframe["identifier"].astype(str)
