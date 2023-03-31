from typing import Any, List, Union, Dict, Iterable

import numpy as np
import pandas as pd
from cached_property import cached_property
from pandas import Series

from plants_sm.data_structures.dataset.dataset import Dataset
from plants_sm.io.commons import FilePathOrBuffer
from plants_sm.mixins.mixins import CSVMixin, ExcelMixin

PLACEHOLDER_FIELD = 'place_holder'


class SingleInputDataset(Dataset, CSVMixin, ExcelMixin):
    _features: Dict[str, Dict[str, np.ndarray]]
    _features_names: List[str]
    _dataframe: Any
    _labels_names: List[str] = None
    _instances: Dict[str, dict]
    _features_fields: Dict[str, Union[str, List[Union[str, int]], slice]]
    _identifiers: List[Union[str, int]] = None

    def __init__(self, dataframe: Any = None, representation_field: str = None,
                 features_fields: Union[str, List[Union[str, int]], slice] = None,
                 labels_field: Union[str, List[Union[str, int]]] = None,
                 instances_ids_field: str = None,
                 batch_size: int = None):
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
        batch_size: int (optional)
            batch size
        """

        # the features fields is a list of fields that are used to extract the features
        # however, they can be both strings or integers, so we need to check the type of the field
        # and convert it to a list of strings
        super().__init__(batch_size=batch_size)
        if dataframe is not None:
            if not isinstance(features_fields, List) and not isinstance(features_fields, slice) and \
                    features_fields is not None:
                self._features_fields = {PLACEHOLDER_FIELD: [features_fields]}
            elif features_fields is not None:
                self._features_fields = {PLACEHOLDER_FIELD: features_fields}
            else:
                self._features_fields = {}

            # the instance ids field is defined here, however, if it is None, eventually is derived from the dataframe
            # setter
            self.instances_ids_field = instances_ids_field

            self.representation_fields = {PLACEHOLDER_FIELD: representation_field}
            self.representation_field = representation_field

            # the dataframe setter will derive the instance ids field if it is None
            # and also will try to set the features names
            self.dataframe = dataframe

            if labels_field is not None:
                if not isinstance(labels_field, List):
                    self._labels_names = [labels_field]
                else:
                    self._labels_names = labels_field

            if self._features_fields:
                self._features = \
                    {PLACEHOLDER_FIELD:
                         self.dataframe.loc[:, self._features_fields[PLACEHOLDER_FIELD]].T.to_dict('list')}
            else:
                self._features = {}

            if self.batch_size is not None:
                while next(self):
                    pass

        # in the case that the dataframe is None and the features field is not None, the features names will be set

    @classmethod
    def from_csv(cls, file_path: FilePathOrBuffer, representation_field: str = None,
                 features_fields: Union[str, List[Union[str, int]], slice] = None,
                 labels_field: Union[str, List[Union[str, int]]] = None,
                 instances_ids_field: str = None, batch_size: Union[None, int] = None,
                 **kwargs) -> 'SingleInputDataset':
        """
        Method to create a dataset from a csv file.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the csv file
        representation_field: str | List[str | int] (optional)
            representation column field (to be processed)
        features_fields: str | List[str | int] | slice (optional)
            features column field
        labels_field: str | List[str | int] (optional)
            labels column field
        instances_ids_field:
            instances column field
        batch_size: int (optional)
            batch size to be used for the dataset
        kwargs:
            additional arguments for the pandas read_csv method

        Returns
        -------
        dataset: SingleInputDataset
            dataset created from the csv file
        """

        instance = cls()
        dataframe = instance._from_csv(file_path, batch_size, **kwargs)
        dataset = SingleInputDataset(dataframe, representation_field,
                                     features_fields, labels_field,
                                     instances_ids_field,
                                     batch_size=batch_size)
        return dataset

    @classmethod
    def from_excel(cls, file_path: FilePathOrBuffer, representation_field: str = None,
                   features_fields: Union[str, List[Union[str, int]], slice] = None,
                   labels_field: Union[str, List[Union[str, int]]] = None,
                   instances_ids_field: str = None, batch_size: Union[None, int] = None, **kwargs) \
            -> 'SingleInputDataset':
        """
        Method to create a dataset from an excel file.

        Parameters
        ----------
        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the csv file
        representation_field: str | List[str | int] (optional)
            representation column field (to be processed)
        features_fields: str | List[str | int] | slice (optional)
            features column field
        labels_field: str | List[str | int] (optional)
            labels column field
        instances_ids_field:
            instances column field
        batch_size: int (optional)
            batch size to be used for the dataset
        kwargs:
            additional arguments for the pandas read_excel method

        Returns
        -------
        dataset: SingleInputDataset
            dataset created from the excel file
        """

        instance = cls()
        dataframe = instance._from_excel(file_path, batch_size=batch_size, **kwargs)
        dataset = SingleInputDataset(dataframe, representation_field,
                                     features_fields, labels_field,
                                     instances_ids_field,
                                     batch_size=batch_size)
        return dataset

    @property
    def identifiers(self) -> List[Union[str, int]]:
        """
        Property for identifiers. It should return the identifiers of the dataset.
        -------
        list of the identifiers: List[Union[str, int]]
        """
        return self._identifiers

    @property
    def features(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Property for features. It should return the features of the dataset.
        """
        if self._features is None:
            raise ValueError('Features are not defined')
        return self._features

    @features.setter
    def features(self, value: Dict[str, Dict[str, np.ndarray]]):
        """
        Setter for features.
        Parameters
        ----------
        value : Dict[str, np.ndarray]
            dictionary of the features
        Returns
        -------
        """
        self._features = value

    def add_features(self, instance_type: str, features: Dict[str, np.ndarray]):
        """
        Add features to the dataset.
        Parameters
        ----------
        instance_type : str
            instance type
        features : Dict[str, np.ndarray]
            dictionary of the features
        Returns
        -------
        """
        if self._features is None:
            self._features = {}
        self._features[instance_type] = features
        self.__dict__.pop('X', None)

    @property
    def features_fields(self):
        return self._features_fields

    @features_fields.setter
    def features_fields(self, value: Dict[str, Union[str, List[Union[str, int]], slice]]):
        """
        Setter for features fields.
        Parameters
        ----------
        value : Union[str, List[Union[str, int]], slice]
            features fields
        Returns
        -------
        """
        self._features_fields = value

    @property
    def labels(self) -> Dict[str, Any]:
        """
        This property will contain the labels for supervised learning.
        Returns
        -------
        Labels for training and prediction. ALIASES: y vector with labels for classification and regression.
        """
        if self._labels_names is not None:
            return self.dataframe.loc[:, self._labels_names].T.to_dict('list')

    @cached_property
    def X(self) -> np.ndarray:
        """
        Property for X. It should return the features of the dataset.
        """
        return np.array(list(self.features[PLACEHOLDER_FIELD].values()))

    @property
    def y(self) -> np.ndarray:
        """
        Alias for the labels property.
        """
        if not self._labels_names:
            raise ValueError('Labels are not defined')
        return np.array(list(self.labels.values()))

    @property
    def instances(self) -> Dict[str, dict]:
        """
        This property will contain the instances of the dataset.
        Returns
        -------
        Array with the instances.
        """
        return self._instances

    def get_instances(self, instance_type: str = PLACEHOLDER_FIELD):
        return self.instances[instance_type]

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
            go = isinstance(value, Iterable) and not isinstance(value, pd.DataFrame) and self.batch_size is not None
            if go:
                # accounting on generators for batch reading
                self._dataframe_generator = value
                value = next(self._dataframe_generator)

            self._set_dataframe(value)
            if self.instances_ids_field is None:
                self._set_instances_ids_field()
            else:
                identifiers = self._dataframe.loc[:, self.instances_ids_field].values
                instances = self.dataframe.loc[:, self.representation_field].values
                self._instances = {PLACEHOLDER_FIELD: dict(zip(identifiers, instances))}
                self.dataframe.drop(self.representation_field, axis=1, inplace=True)

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

    def _set_instances_ids_field(self):
        """
        Private method to set the instances ids field if it is not defined.
        """
        if self.instances_ids_field is None:
            self.instances_ids_field = "identifier"
            identifiers_series = Series(list(range(self.dataframe.shape[0])), name="identifier")

            if self._features_fields:
                if isinstance(self._features_fields[PLACEHOLDER_FIELD], slice):
                    features_fields_slice = self._features_fields[PLACEHOLDER_FIELD]
                    if features_fields_slice.start is not None:
                        start = features_fields_slice.start
                    else:
                        start = 0

                    if features_fields_slice.stop is not None:
                        stop = features_fields_slice.stop
                    else:
                        stop = self._dataframe.columns.size

                    if features_fields_slice.step is not None:
                        step = features_fields_slice.step
                    else:
                        step = 1

                    indexes_list = list(range(start, stop, step))
                    self._features_fields = {PLACEHOLDER_FIELD: [self._dataframe.columns[i] for i in indexes_list]}

                elif isinstance(self._features_fields[PLACEHOLDER_FIELD][0], int):
                    self._features_fields[PLACEHOLDER_FIELD] = [self._dataframe.columns[i] for i in
                                                                self._features_fields[PLACEHOLDER_FIELD]]

            self._dataframe = pd.concat((identifiers_series, self._dataframe), axis=1)
            self._dataframe["identifier"] = self._dataframe["identifier"].astype(str)
            self._identifiers = self._dataframe["identifier"].values

            instances = self.dataframe.loc[:, self.representation_field].values
            self._instances = {PLACEHOLDER_FIELD: dict(zip(identifiers_series.values, instances))}
            self.dataframe.drop(self.representation_field, axis=1, inplace=True)
        else:
            self._identifiers = self._dataframe[self.instances_ids_field].values