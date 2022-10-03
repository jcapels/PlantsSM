import functools
from abc import ABCMeta, abstractmethod
from typing import Any, List, Union, Dict

import numpy as np
import pandas as pd
from pandas import Series

from plants_sm.data_structures._utils import NUMERICAL_TYPES
from plants_sm.mixins.mixins import CSVMixin, ExcelMixin


class Dataset(metaclass=ABCMeta):

    def __init__(self, dataframe: Any,
                 representation_field: Union[str, List[Union[str, int]]] = None,
                 features_fields: Union[str, List[Union[str, int]], slice] = None,
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
        features_fields: str | List[str | int] | slice (optional)
            features column field
        labels_field: str | List[str | int] (optional)
            labels column field
        instances_ids_field: str | List[str | int] (optional)
            instances column field
        """

        # the features fields is a list of fields that are used to extract the features
        # however, they can be both strings or integers, so we need to check the type of the field
        # and convert it to a list of strings
        self._features_shape = None
        self._features_dataframe = None
        self._n_dimensional_features_array = None
        if not isinstance(features_fields, List) and not isinstance(features_fields, slice) and \
                features_fields is not None:
            self._features_fields = [features_fields]
        else:
            self._features_fields = features_fields

        # the instance ids field is defined here, however, if it is None, eventually is derived from the dataframe
        # setter
        self.instances_ids_field = instances_ids_field
        self._n_dimensional_features = {}

        # the dataframe setter will derive the instance ids field if it is None
        # and also will try to set the features names
        self._features_names = None
        self._dataframe = None
        self.dataframe = dataframe

        self._representation_field = None
        self.representation_field = representation_field

        self._labels_names = None
        if not isinstance(labels_field, List):
            self.labels_names = [labels_field]
        else:
            self.labels_names = labels_field

        # in the case that the dataframe is None and the features field is not None, the features names will be set

    @property
    def features_fields(self) -> Union[List[Any], slice]:
        """
        Property for features names
        Returns
        -------
        list of the names of the features
        """
        return self._features_fields

    @features_fields.setter
    def features_fields(self, value: List[Any]):
        """
        Setter for features names.
        Parameters
        ----------
        value : List[Any]
            list of the features names
        Returns
        -------
        """
        if isinstance(value, List) or isinstance(value, slice):
            self._features_fields = value

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
    def features(self) -> np.ndarray:
        """
        This property will be important for retrieving the columns of the dataframe with the features.

        Returns
        -------

        Features taken from the dataframe. ALIASES: X matrix/vector/array with the features.

        """
        pass

    @property
    def X(self) -> np.ndarray:
        """
        Alias for the features property.
        """
        return self.features

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
    def labels(self) -> np.ndarray:
        """
        This property will contain the labels for supervised learning.
        Returns
        -------
        Labels for training and prediction. ALIASES: y vector with labels for classification and regression.
        """
        pass

    @property
    def y(self) -> np.ndarray:
        """
        Alias for the labels property.
        """
        return self.labels

    @property
    @abstractmethod
    def instances(self) -> np.ndarray:
        """
        This property will contain the instances of the dataset.
        Returns
        -------
        Array with the instances.
        """
        pass

    @property
    def features_shape(self) -> tuple:
        """
        Returns
        -------
        shape of the features array
        """
        return self._features_shape

    @features_shape.setter
    def features_shape(self, value: tuple):
        """
        Parameters
        ----------
        value: tuple
            shape of the features array
        Returns
        -------
        """
        self._features_shape = value

    @property
    def features_dataframe(self) -> pd.DataFrame:
        """
        This property will contain the features dataframe.

        Returns
        -------
        """
        return self._features_dataframe

    @features_dataframe.setter
    def features_dataframe(self, value: pd.DataFrame):
        """
        This property will contain the features dataframe.

        Parameters
        ----------
        value: pd.DataFrame
            dataframe with the features

        Returns
        -------

        """
        self._features_dataframe = value

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
            if self.instances_ids_field is None:
                self._set_instances_ids_field()
            else:
                self._dataframe.set_index(self.instances_ids_field, inplace=True)
                self._dataframe.sort_index(inplace=True)

            if self.features_fields is not None:

                try:
                    self.features_dataframe = self.dataframe.loc[:, self.features_fields]
                    self.features_shape = self.features_dataframe.shape
                except (TypeError, KeyError):
                    self.features_dataframe = self.dataframe.iloc[:, self.features_fields]
                    self.features_shape = self.features_dataframe.shape

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
    def representation_field(self) -> Union[str, int, Dict[str, str]]:
        """
        Property of the representation of the molecule, reaction or compounds
        Returns
        -------
        Representation field: str | int | Dict[str, str]
            Field where the biological entity is represented. It can also be a dictionary with the fields
        Examples
        --------
        >>> dataset.representation_field = {"protein": "protein_field", "ligand": "ligand_field"}
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

    @abstractmethod
    def drop_nan(self, fields: List[Union[str, int]] = None, **kwargs) -> 'Dataset':
        """
        Private method to drop the nan values from the dataframe.
        Parameters
        ----------
        fields: List[Union[str, int]]
            list of the fields to drop the nan values
        Returns
        -------
        Dataset: Dataset
            dataset without the nan values
        """
        pass


class PandasDataset(Dataset, CSVMixin, ExcelMixin):

    def __init__(self, dataframe: pd.DataFrame = None,
                 representation_field: Union[str, List[Union[str, int]]] = None,
                 features_fields: Union[str, List[Union[str, int]], slice] = None,
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
        features_fields: str | List[str | int] | slice (optional)
            features column field
        labels_field: str | List[str | int] (optional)
            labels column field
        instances_ids_field: str | List[str | int] (optional)
            instances column field
        """

        super().__init__(dataframe, representation_field, features_fields, labels_field, instances_ids_field)

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
    def features(self) -> np.ndarray:
        """
        This property will only go to the dataframe and return a chunk with features.
        Returns
        -------
        features : array with the features
        """
        if self.features_fields is not None:

            if len(self.features_shape) <= 2:
                features = self.features_dataframe.loc[:, self.features_fields].to_numpy()
                assert features.shape[1] == self.features_shape[1], \
                    "The number of features is not the same as the number of features fields"
                return features

            elif len(self.features_shape) == 3:
                features = self.features_dataframe.loc[:, self.features_fields].to_numpy() \
                    .reshape((len(self.identifiers), self.features_shape[1], len(self.features_fields)))

                assert features.shape[1] == self.features_shape[1], \
                    "The number of features is not the same as the number of features fields"

                return features

        else:
            raise ValueError("The features were not extracted yet.")

    @property
    def labels(self) -> np.ndarray:
        """
        This property will only go to the dataframe and return a chunk with labels.
        Returns
        -------
        """
        return np.array(self.dataframe.loc[:, self.labels_names])

    @property
    def instances(self) -> np.ndarray:
        """
        This property will only go to the dataframe and return a chunk with instances.
        Returns
        -------
        """
        return np.array(self.dataframe.loc[:, self.representation_field])

    @functools.cached_property
    def identifiers(self) -> List[str]:
        """
        This property will only go to the dataframe and return a list with the identifiers.
        Returns
        -------
        """
        return self.dataframe.index.tolist()

    def _set_instances_ids_field(self):
        """
        Private method to set the instances ids field if it is not defined. If not defined, it is necessary to
        increment one field to the features fields if they are integers or slices.
        """
        if self.instances_ids_field is None:
            self.instances_ids_field = "identifier"
            identifiers_series = Series(list(range(self.dataframe.shape[0])), name="identifier")

            if self.features_fields is not None:
                if isinstance(self.features_fields, slice):
                    if self.features_fields.start is not None:
                        start = self.features_fields.start
                    else:
                        start = 0

                    if self.features_fields.stop is not None:
                        stop = self.features_fields.stop
                    else:
                        stop = self._dataframe.columns.size

                    if self.features_fields.step is not None:
                        step = self.features_fields.step
                    else:
                        step = 1

                    indexes_list = list(range(start, stop, step))
                    self.features_fields = [self._dataframe.columns[i] for i in indexes_list]

                elif isinstance(self.features_fields[0], int):
                    self.features_fields = [self._dataframe.columns[i] for i in self.features_fields]

            self._dataframe = pd.concat((identifiers_series, self._dataframe), axis=1)
            self._dataframe["identifier"] = self._dataframe["identifier"].astype(str)
            self._dataframe.set_index("identifier", inplace=True)

    def drop_nan(self, fields: List[Union[str, int]] = None, **kwargs) -> 'PandasDataset':
        """
        Method to drop the nan values from the dataframe.
        Parameters
        ----------
        fields: List[Union[str, int]]
            list of the fields to drop the nan values
        Returns
        -------
        PandasDataset: PandasDataset
            pandas dataset without the nan values
        """
        drop_columns = False
        if "inplace" not in kwargs:
            inplace = True
            kwargs["inplace"] = inplace

        if "axis" in kwargs:
            axis = kwargs["axis"]
            if axis == 1:
                drop_columns = True

        if not fields:
            if drop_columns:
                bool_idx = self.dataframe.isnull().any(axis=0)
                bool_idx = bool_idx.values
                columns_to_remove = self.dataframe.columns[bool_idx].values

                if self.representation_field in columns_to_remove:
                    raise ValueError("You dropped the representation field of the dataframe, no changes were made.")
                elif any(label in columns_to_remove for label in self.labels_names):
                    raise ValueError("You dropped one of the labels field of the dataframe, no changes were made.")
                elif self.instances_ids_field in columns_to_remove:
                    raise ValueError("You dropped the instances ids field of the dataframe, no changes were made.")

                for column in columns_to_remove:
                    self.features_fields.remove(column)

            self.dataframe.dropna(**kwargs)

        else:
            if drop_columns:

                if isinstance(fields[0], int):
                    fields = [self.dataframe.columns[field] for field in fields]
                if self.representation_field in fields:
                    raise ValueError("You dropped the representation field of the dataframe, no changes were made.")
                elif any(label in fields for label in self.labels_names):
                    raise ValueError("You dropped one of the labels field of the dataframe, no changes were made.")
                elif self.instances_ids_field in fields:
                    raise ValueError("You dropped the instances ids field of the dataframe, no changes were made.")

                bool_idx = self.dataframe.isnull().any(axis=0)
                bool_idx = bool_idx.values
                columns_to_remove = self.dataframe.columns[bool_idx].values
                for column in columns_to_remove:
                    self.features_fields.remove(column)

            self.dataframe.dropna(subset=fields, **kwargs)
        return self
