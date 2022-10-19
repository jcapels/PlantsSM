from functools import cached_property
from typing import List, Any, Union, Dict

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
    _labels_names: List[str]
    _features_fields: Dict[str, Union[str, int]]
    _instances: Dict[str, Dict[str, Any]]
    _identifiers: List[Union[str, int]]

    def __init__(self, dataframe: Any = None, representation_fields: Dict[str, Union[str, int]] = None,
                 labels_field: Union[str, List[Union[str, int]]] = None,
                 instances_ids_field: Dict[str, Union[str, int]] = None):
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
        """

        # the features fields is a list of fields that are used to extract the features
        # however, they can be both strings or integers, so we need to check the type of the field
        # and convert it to a list of strings
        super().__init__()
        if dataframe is not None:
            self.instances_ids_field = instances_ids_field

            # the dataframe setter will derive the instance ids field if it is None
            # and also will try to set the features names
            self._dataframe = dataframe

            self.representation_field = representation_fields

            if labels_field is not None:
                if not isinstance(labels_field, List):
                    self.labels_names = [labels_field]
                else:
                    self.labels_names = labels_field

                self._labels = self.dataframe.loc[:, self.labels_names].T.to_dict('list')
            else:
                self._labels = None

            self.set_instances_and_ids()

    @property
    def dataframe(self) -> Any:
        return self._dataframe

    @classmethod
    def from_csv(cls, file_path: FilePathOrBuffer, representation_fields: Dict[str, Union[str, int]] = None,
                 labels_field: Union[str, List[Union[str, int]]] = None,
                 instances_ids_field: Dict[str, Union[str, int]] = None, **kwargs) -> 'MultiInputDataset':

        instance = cls()
        dataframe = instance._from_csv(file_path, **kwargs)
        dataset = MultiInputDataset(dataframe, representation_fields,
                                    labels_field, instances_ids_field)
        return dataset

    @classmethod
    def from_excel(cls, file_path: FilePathOrBuffer,
                   representation_fields: Dict[str, Union[str, int]] = None,
                   labels_field: Union[str, List[Union[str, int]]] = None,
                   instances_ids_field: Dict[str, Union[str, int]] = None, **kwargs) -> 'MultiInputDataset':

        instance = cls()
        dataframe = instance._from_excel(file_path, **kwargs)
        dataset = MultiInputDataset(dataframe, representation_fields,
                                    labels_field, instances_ids_field)
        return dataset

    @property
    def instances(self):
        return self._instances

    @property
    def identifiers(self):
        """
        The identifiers of the instances (interaction between the instances)
        """
        return self.dataframe.index.values

    @property
    def features_fields(self):
        return self._features_fields

    def get_instances(self, instance_type):
        return self._instances[instance_type]

    def set_instances_and_ids(self):
        self._instances = {}
        instance_types = []
        for instance_type, instance_field in self.representation_field.items():
            if instance_type not in self.instances_ids_field:
                self.generate_ids(instance_type)

            identifiers = self.dataframe.loc[:, self.instances_ids_field[instance_type]].values
            instances = self.dataframe.loc[:, instance_field].values

            self._instances[instance_type] = dict(zip(identifiers, instances))
            instance_types.append(instance_type)

        for id_type, _ in self.instances_ids_field.items():
            if id_type not in instance_types:
                self.dataframe.set_index(self.instances_ids_field[id_type], inplace=True)

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

    @cached_property
    def X(self):
        pass

    @property
    def y(self):
        pass
