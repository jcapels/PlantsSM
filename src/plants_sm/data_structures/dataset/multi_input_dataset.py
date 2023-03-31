from cached_property import cached_property
from typing import List, Any, Union, Dict, Iterable

import numpy as np
import pandas as pd

from plants_sm.data_structures.dataset import Dataset
from plants_sm.io.commons import FilePathOrBuffer
from plants_sm.mixins.mixins import CSVMixin, ExcelMixin


class MultiInputDataset(Dataset, CSVMixin, ExcelMixin):
    _features: Dict[str, Dict[str, np.ndarray]]
    _features_names: List[str]
    _dataframe: pd.DataFrame
    _representation_field: str
    _labels_names: List[str] = None
    _features_fields: Dict[str, Union[str, int]] = {}
    _instances: Dict[str, Dict[str, Any]]
    _identifiers: List[Union[str, int]] = None

    def __init__(self, dataframe: Any = None, representation_fields: Dict[str, Union[str, int]] = None,
                 labels_field: Union[str, List[Union[str, int]]] = None,
                 instances_ids_field: Dict[str, Union[str, int]] = None,
                 batch_size: int = None):
        """
        Constructor
        Parameters
        ----------
        dataframe: Any
            dataframe to be consumed by the class and defined as class property
        representation_fields: str | List[str | int] (optional)
            representation column field (to be processed)
        labels_field: str | List[str | int] (optional)
            labels column field
        instances_ids_field: str | List[str | int] (optional)
            instances column field
        batch_size: int (optional)
            the size of the batches
        """

        # the features fields is a list of fields that are used to extract the features
        # however, they can be both strings or integers, so we need to check the type of the field
        # and convert it to a list of strings
        super().__init__(batch_size=batch_size)
        if dataframe is not None:
            self._features = {}
            self._labels = None
            self.instances_ids_field = instances_ids_field
            self.representation_field = representation_fields

            # the dataframe setter will derive the instance ids field if it is None
            # and also will try to set the features names
            self.dataframe = dataframe

            if labels_field is not None:
                if not isinstance(labels_field, List):
                    self.labels_names = [labels_field]
                else:
                    self.labels_names = labels_field

            if self.batch_size is not None:
                while next(self):
                    pass

    @property
    def dataframe(self) -> Any:
        return self._dataframe

    @classmethod
    def from_csv(cls, file_path: FilePathOrBuffer, representation_fields: Dict[str, Union[str, int]] = None,
                 labels_field: Union[str, List[Union[str, int]]] = None,
                 instances_ids_field: Dict[str, Union[str, int]] = None,
                 batch_size: Union[None, int] = None,
                 **kwargs: object) -> 'MultiInputDataset':

        instance = cls()
        dataframe = instance._from_csv(file_path, batch_size, **kwargs)
        dataset = MultiInputDataset(dataframe, representation_fields,
                                    labels_field, instances_ids_field,
                                    batch_size=batch_size)
        return dataset

    @classmethod
    def from_excel(cls, file_path: FilePathOrBuffer,
                   representation_fields: Dict[str, Union[str, int]] = None,
                   labels_field: Union[str, List[Union[str, int]]] = None,
                   instances_ids_field: Dict[str, Union[str, int]] = None,
                   batch_size: Union[None, int] = None, **kwargs) -> 'MultiInputDataset':

        instance = cls()
        dataframe = instance._from_excel(file_path, batch_size, **kwargs)
        dataset = MultiInputDataset(dataframe, representation_fields,
                                    labels_field, instances_ids_field,
                                    batch_size=batch_size)
        return dataset

    @property
    def instances(self):
        return self._instances

    @property
    def identifiers(self):
        """
        The identifiers of the instances (interaction between the instances)
        """
        return self._identifiers

    @identifiers.setter
    def identifiers(self, value: List[Union[str, int]]):
        """
        Setter of the property. It verified the type of the value inputted.

        Parameters
        ----------
        value: List[Union[str, int]]
            identifiers to be set

        """
        self._identifiers = value

    @property
    def labels(self):
        return self._labels

    @property
    def features_fields(self):
        return self._features_fields

    def get_instances(self, instance_type: str = None):
        if instance_type is None:
            raise ValueError("The instance type must be specified")
        return self._instances[instance_type]

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
            self.set_instances_and_ids()

    def set_instances_and_ids(self):
        """
        Set the instances and the identifiers of the instances
        """
        self._instances = {}
        instance_types = []
        for instance_type, instance_field in self.representation_field.items():
            if instance_type not in self.instances_ids_field:
                self.generate_ids(instance_type)
            elif self.instances_ids_field[instance_type] not in self.dataframe.columns:
                self.generate_ids(instance_type)

            identifiers = self.dataframe.loc[:, self.instances_ids_field[instance_type]].values
            instances = self.dataframe.loc[:, instance_field].values

            self._instances[instance_type] = dict(zip(identifiers, instances))
            instance_types.append(instance_type)

        if self._identifiers is None:
            _identifiers_ids = []
            for id_type, ids_field in self.instances_ids_field.items():
                if id_type not in instance_types:
                    _identifiers_ids.append(ids_field)

            self._identifiers = self.dataframe.loc[:, _identifiers_ids].values
        self.dataframe.drop(list(self.representation_field.values()), axis=1, inplace=True)

    def generate_ids(self, instance_type):
        """
        Generates a new identifiers for the instance type
        Parameters
        ----------
        instance_type: str
            the instance type
        Returns
        -------
        int
            the new identifier
        """
        # generate identifiers for unique elements
        unique_elements = np.unique(self.dataframe.loc[:, self.representation_field[instance_type]].values)
        new_ids = np.arange(1, len(unique_elements) + 1)

        self.dataframe.loc[:, f"{instance_type}_identifiers"] = \
            self.dataframe.loc[:, self.representation_field[instance_type]].map(dict(zip(unique_elements, new_ids)))

        self.instances_ids_field[instance_type] = f"{instance_type}_identifiers"

    @property
    def features(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Property for features. It should return the features of the dataset.
        """
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
        self.__dict__.pop('X', None)

    def add_features(self, instance_type: str, features: Dict[str, np.ndarray]):
        """
        Adds features to the dataset
        Parameters
        ----------
        instance_type: str
            the instance type
        features: Dict[str, np.ndarray]
            the features
        Returns
        -------
        """
        if instance_type in self._features:
            self._features[instance_type].update(features)
        else:
            self._features[instance_type] = features

        self.__dict__.pop('X', None)

    def to_csv(self, file_path: FilePathOrBuffer, **kwargs) -> bool:
        """
        Saves the dataset to a csv file
        Parameters
        ----------
        file_path: FilePathOrBuffer
            the path of the file
        kwargs: dict
            the arguments for the pandas.to_csv method
        Returns
        -------
        bool
            True if the operation is successful
        """
        new_dataframe = self.dataframe.copy()
        for instance_type in self.instances.keys():
            new_dataframe.loc[:, instance_type] = new_dataframe.loc[:, instance_type].map(dict(zip(self.identifiers,
                                                                                                   self.identifiers)))
        try:
            instances = self.X.keys()
        except ValueError:
            instances = None

        if self._features_fields is not None and instances is not None:
            for instance in instances:
                instance_features = self.X[instance]
                new_dataframe.loc[:, self._features_fields[instance]] = instance_features

        if self._labels_names is not None:
            new_dataframe.loc[:, self._labels_names] = self.y

        new_dataframe.to_csv(file_path, **kwargs)

    @cached_property
    def X(self):
        """
        Property for X. It should return the features of the dataset.

        Returns
        -------
        Dict[str, np.ndarray]
            the features, the key is the instance type and the value is the features
        """

        res = {}
        feature_keys = self.features.keys()
        for instance_type, instance in self.instances.items():  # probably 2 or a little more instances
            res[instance_type] = []

            if instance_type in feature_keys:
                instance_ids = self.dataframe.loc[:, self.instances_ids_field[instance_type]]
                for instance_id in instance_ids:
                    res[instance_type].append(self.features[instance_type][instance_id])

                res[instance_type] = np.array(res[instance_type])
            else:
                raise ValueError(f"Features for {instance_type} are not defined")
        return res

    @property
    def y(self):
        if self.labels_names is not None:
            return np.array(self.dataframe.loc[:, self.labels_names].values)
