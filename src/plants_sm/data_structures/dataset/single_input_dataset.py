import time
import warnings
from typing import Any, Iterable, List, Union, Dict

import numpy as np
import pandas as pd
from cached_property import cached_property
from pandas import Series

from plants_sm.data_structures.dataset.dataset import Dataset
from plants_sm.io import write_csv
from plants_sm.io.commons import FilePathOrBuffer
from plants_sm.io.pickle import write_pickle
from plants_sm.mixins.mixins import CSVMixin, ExcelMixin
from plants_sm.data_structures.dataset._utils import process_slices

PLACEHOLDER_FIELD = 'place_holder'


class SingleInputDataset(Dataset, CSVMixin, ExcelMixin):
    _features: Dict[str, Dict[str, np.ndarray]]
    _features_names: List[str]
    _dataframe: Any
    _labels_names: Union[List[str], None] = None
    _instances: Dict[str, dict]
    _features_fields: Dict[str, Union[str, List[Union[str, int]], slice]]
    _identifiers: Union[List[Union[str, int]], None] = None
    _automatically_generated_identifiers: bool = False

    def __init__(self, dataframe: Any = None, representation_field: Union[str, None] = None,
                 features_fields: Union[str, List[Union[str, int]], slice, None] = None,
                 labels_field: Union[str, List[Union[str, int]], slice, None] = None,
                 instances_ids_field: Union[str, None] = None,
                 batch_size: Union[int, None] = None):
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

            elif isinstance(features_fields, slice):

                indexes_list = process_slices(dataframe.columns, features_fields)
                features_fields = [dataframe.columns[i] for i in indexes_list]

                self._features_fields = {PLACEHOLDER_FIELD: features_fields}

            elif features_fields is not None:
                self._features_fields = {PLACEHOLDER_FIELD: features_fields}
            else:
                self._features_fields = {}

            # the instance ids field is defined here, however, if it is None, eventually is derived from the dataframe
            # setter
            self.instances_ids_field = instances_ids_field

            self.representation_field = representation_field

            # the dataframe setter will derive the instance ids field if it is None
            # and also will try to set the features names
            self.dataframe = dataframe
            self._identifiers = self.dataframe[self.instances_ids_field].values
            self.dataframe.index = self.identifiers
            if labels_field is not None:

                if isinstance(labels_field, slice):
                    indexes_list = process_slices(self._dataframe.columns, labels_field)
                    self._labels_names = [self._dataframe.columns[i] for i in indexes_list]

                elif isinstance(labels_field, list):

                    if isinstance(labels_field[0], int):
                        self._labels_names = [self._dataframe.columns[i] for i in
                                              labels_field]

                else:
                    self._labels_names = [labels_field]

                self._labels = None
                # self._labels = self.dataframe.loc[:, self._labels_names].T.to_dict('list')
            else:
                self._labels = None

            if self._features_fields:

                features = self.dataframe.loc[:, self._features_fields[PLACEHOLDER_FIELD]]

                self._features = {PLACEHOLDER_FIELD: {k: features.iloc[i, :].values
                                                      for i, k in enumerate(self._identifiers)}}


            else:
                self._features = {}

            # set the index of the dataframe to the instances ids field
            # self._dataframe.set_index(self.instances_ids_field, inplace=True)

            self.dataframe.drop(self.representation_field, axis=1, inplace=True)
            if self.batch_size is not None:
                while next(self):
                    pass

                self.next_batch()

        # in the case that the dataframe is None and the features field is not None, the features names will be set
    
    def __len__(self):
        return len(self.instances[PLACEHOLDER_FIELD])

    def __getitem__(self, idx):
        return self.instances[PLACEHOLDER_FIELD][idx], self.dataframe.loc[idx, self._labels_names].values

    @classmethod
    def from_csv(cls, file_path: FilePathOrBuffer, representation_field: Union[str, None] = None,
                 features_fields: Union[str, List[Union[str, int]], slice, None] = None,
                 labels_field: Union[str, List[Union[str, int]], slice, None] = None,
                 instances_ids_field: Union[str, None] = None,
                 batch_size: Union[None, int] = None,
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

    def to_csv(self, file_path: FilePathOrBuffer, **kwargs):
        """
        Method to write the dataset to a csv file.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the csv file
        kwargs:
            additional arguments for the pandas to_csv method

        Returns
        -------
        success: bool
            True if the file was written successfully, False otherwise
        """

        new_dataframe = self.dataframe.copy()
        data = list(self.instances[PLACEHOLDER_FIELD].items())
        data = pd.DataFrame(data, columns=[self.instances_ids_field, self.representation_field])

        new_dataframe = pd.merge(new_dataframe, data, on=self.instances_ids_field, how='left')

        if self.features:
            write_pkl = False
            features = list(self.features[PLACEHOLDER_FIELD].values())[0]
            if features.ndim > 1:
                warnings.warn(f"The features are not 2D, writing to pickle file")
                write_pkl = True

            if not write_pkl:
                # convert a dictionary into a list of tuples
                data = [(k, *v.tolist()) for k, v in self.features[PLACEHOLDER_FIELD].items()]
                data = pd.DataFrame(data, columns=[self.instances_ids_field] +
                                                  self.features_fields[PLACEHOLDER_FIELD])

                new_dataframe = pd.merge(new_dataframe, data, on=self.instances_ids_field, how='left')

            else:
                write_pickle(file_path.replace("csv", "pkl"), self.features)

        write_csv(file_path, new_dataframe, **kwargs)

    @classmethod
    def from_excel(cls, file_path: FilePathOrBuffer, representation_field: Union[str, None] = None,
                   features_fields: Union[str, List[Union[str, int]], slice, None] = None,
                   labels_field: Union[str, List[Union[str, int]], slice, None] = None,
                   instances_ids_field: Union[str, None] = None,
                   batch_size: Union[int, None] = None, **kwargs) \
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
        self._clear_cached_properties()
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
        if self.features == {}:
            raise ValueError('Features are not defined')
        features = [self.features["place_holder"][sequence_id] for sequence_id in
                    self.dataframe[self.instances_ids_field]]
        try:
            return np.array(features, dtype=np.float32)
        except ValueError:
            warnings.warn("The features are not 2D and different dimensions in one of the shapes, "
                          "returning a list of arrays")
            return np.array(features)

    @cached_property
    def y(self) -> np.ndarray:
        """
        Alias for the labels property.
        """
        if not self._labels_names:
            raise ValueError('Labels are not defined')
        # return np.array(list(self.labels.values()))
        return self.dataframe.loc[:, self._labels_names].to_numpy()

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
                # self.dataframe.drop(self.representation_field, axis=1, inplace=True)

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
            identifiers_series = Series(list(map(str, range(self.dataframe.shape[0]))), name="identifier", dtype=str)

            if self._features_fields:
                if isinstance(self._features_fields[PLACEHOLDER_FIELD], slice):
                    features_fields_slice = self._features_fields[PLACEHOLDER_FIELD]

                    indexes_list = process_slices(self._dataframe.columns, features_fields_slice)

                    self._features_fields = {PLACEHOLDER_FIELD: [self._dataframe.columns[i] for i in indexes_list]}

                elif isinstance(self._features_fields[PLACEHOLDER_FIELD][0], int):
                    self._features_fields[PLACEHOLDER_FIELD] = [self._dataframe.columns[i] for i in
                                                                self._features_fields[PLACEHOLDER_FIELD]]

            self._automatically_generated_identifiers = True

            self._dataframe = pd.concat((identifiers_series, self._dataframe), axis=1)
            self._dataframe["identifier"] = self._dataframe["identifier"]
            self._identifiers = self._dataframe["identifier"].values

            instances = self.dataframe.loc[:, self.representation_field].values
            self._instances = {PLACEHOLDER_FIELD: dict(zip(identifiers_series.values, instances))}
            # self.dataframe.drop(self.representation_field, axis=1, inplace=True)
        else:
            self._identifiers = self._dataframe[self.instances_ids_field].values

    def select(self, ids: Union[List[str], List[int]], instance_type: str = PLACEHOLDER_FIELD):
        """
        Select a subset of the dataset based on the identifiers.

        Parameters
        ----------
        ids : Union[List[str], List[int]]
            list of identifiers to be selected
        instance_type : str
            type of the instances to be selected
        """

        if self.instances_ids_field is None:
            raise ValueError("Instances ids field is not defined")

        self._dataframe = self._dataframe[self._dataframe[self.instances_ids_field].isin(ids)]
        self._identifiers = self._dataframe[self.instances_ids_field].values

        for instance_type in self._instances:
            self._instances[instance_type] = {k: v for k, v in self._instances[instance_type].items() if k in ids}

            if self._features:
                self._features[instance_type] = {k: v for k, v in self._features[instance_type].items() if k in ids}

        self._clear_cached_properties()

    def merge(self, dataset: 'SingleInputDataset'):

        if self._automatically_generated_identifiers and dataset._automatically_generated_identifiers:
            dataset._identifiers = [f"{i}_" for i in dataset._identifiers]
            # change the identifiers in the dataset instances and features
            dataset._instances[PLACEHOLDER_FIELD] = {k + "_": v for k, v in dataset._instances[PLACEHOLDER_FIELD].items()}

            if self._features:
                dataset._features[PLACEHOLDER_FIELD] = {k + "_": v for k, v in dataset._features[PLACEHOLDER_FIELD].items()}

            dataset._dataframe[self.instances_ids_field] = (
                dataset._dataframe[self.instances_ids_field].apply(lambda x: x + "_"))

        # elif set(self._identifiers).intersection(set(dataset._identifiers)):
        #     raise ValueError("The datasets have common identifiers")

        self._dataframe = pd.concat((self._dataframe, dataset.dataframe), axis=0)

        for instance_type in self._instances:
            self._instances[instance_type].update(dataset._instances[instance_type])

            if self._features:
                self._features[instance_type].update(dataset._features[instance_type])

        self._clear_cached_properties()

        return self
