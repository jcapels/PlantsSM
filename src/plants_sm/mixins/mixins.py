import os.path
import re
from abc import abstractmethod
from typing import Any, Union, Iterable

import pandas as pd

from plants_sm.io import read_csv, write_csv
from plants_sm.io.commons import FilePathOrBuffer
from plants_sm.io.excel import write_excel, read_excel
from plants_sm.io.pickle import write_pickle, read_pickle


class DictMixin:
    def to_dict(self):
        return self._traverse_dict(self.__dict__)

    def _traverse_dict(self, attributes):
        result = {}
        for key, value in attributes.items():
            transverse_value = self._traverse(key, value)
            if transverse_value is not None:
                result[key] = transverse_value

        return result

    def _traverse(self, key, value):
        if not re.match("^_", key):
            if isinstance(value, DictMixin):
                return value.to_dict()
            elif isinstance(value, dict):
                return self._traverse_dict(value)
            elif isinstance(value, list):
                return [self._traverse(key, v) for v in value]
            elif hasattr(value, '__dict__'):
                return self._traverse_dict(value.__dict__)
            else:
                return value
        else:
            return None


class PickleMixin:
    def to_pickle(self, file_path: FilePathOrBuffer) -> bool:
        """
        Method to export the object to pickle.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the object will be exported.

        Returns
        -------
        bool: True if the operation was successful, False otherwise.
        """
        return write_pickle(file_path, self)

    @classmethod
    def from_pickle(cls, file_path: FilePathOrBuffer) -> 'PickleMixin':
        """
        Method to import the object from pickle.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the object will be imported.

        Returns
        -------
        PickleMixin: object imported from pickle.
        """
        return read_pickle(file_path)


class CSVMixin:

    def __init__(self):
        self._dataframe = None

    @property
    @abstractmethod
    def dataframe(self) -> Any:
        """
        Abstract method and property that returns the dataframe.
        Returns
        -------

        """
        pass

    @dataframe.setter
    def dataframe(self, value: Any):
        """
        Setter for dataframe.

        Parameters
        ----------
        value: Any
            value to be set as dataframe
        """
        self._dataframe = value

    def to_csv(self, file_path: FilePathOrBuffer, **kwargs) -> bool:
        """
        Method to export the dataframe to csv.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the dataframe will be exported.

        Returns
        -------
        bool: True if the operation was successful, False otherwise.

        Raises
        ------
        NotImplementedError
            If the dataframe is not a type where the package cannot export it.

        """

        if isinstance(self.dataframe, pd.DataFrame):
            return write_csv(filepath_or_buffer=file_path,
                             df=self.dataframe,
                             **kwargs)
        else:
            raise NotImplementedError("This method is not implemented for this type of object")

    @staticmethod
    def retrieve_a_generator(file_path: str, batch_size: Union[None, int], **kwargs) -> Iterable:
        """
        Method to retrieve a generator from a csv file.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the dataframe will be imported.
        batch_size: Union[None, int]
            size of the batch to be read from the csv file.
        kwargs: dict
            arguments to be passed to the read_csv method.

        Returns
        -------
        Iterable: generator that will return the batches of the dataframe.
        """
        gen = read_csv(file_path, chunksize=batch_size, iterator=True, get_buffer=False, **kwargs)
        for batch in gen:
            yield batch

    @staticmethod
    def _from_csv(file_path: FilePathOrBuffer, batch_size: Union[None, int] = None, **kwargs) -> Any:
        """
        Method to import the dataframe from csv.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the dataframe will be imported.
        batch_size: Union[None, int]
            size of the batch to be read from the csv file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        if not batch_size:
            df = read_csv(file_path, **kwargs)
            return df

        else:
            return CSVMixin.retrieve_a_generator(file_path, batch_size, **kwargs)

    @classmethod
    def from_csv(cls, file_path: FilePathOrBuffer, **kwargs) -> 'CSVMixin':
        """
        Method to import the dataframe from csv.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the dataframe will be imported.

        Returns
        -------
        CSVMixin: object of the class that inherits from CSVMixin

        """


class ExcelMixin:

    def __init__(self):
        self._dataframe = None

    @property
    @abstractmethod
    def dataframe(self) -> Any:
        """
        Abstract method and property that returns the dataframe.
        Returns
        -------

        """
        pass

    @dataframe.setter
    def dataframe(self, value: Any):
        """
        Setter for dataframe.

        Parameters
        ----------
        value: Any
            value to be set as dataframe
        """
        self._dataframe = value

    def to_excel(self, file_path: FilePathOrBuffer, **kwargs) -> bool:
        """
        Method to export the dataframe to excel.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the dataframe will be exported.

        Returns
        -------
        bool: True if the file was created, False otherwise.

        """

        if isinstance(self.dataframe, pd.DataFrame):
            return write_excel(file_path, self.dataframe, **kwargs)
        else:
            raise NotImplementedError("This method is not implemented for this type of object")

    @staticmethod
    def retrieve_a_generator(file_path, batch_size, **kwargs) -> Iterable:
        """
        Method to retrieve a generator from a csv file.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the dataframe will be imported.
        batch_size: Union[None, int]
            size of the batch to be read from the csv file.
        kwargs: dict
            arguments to be passed to the read_csv method.

        Returns
        -------
        Iterable: generator that will return the batches of the dataframe.
        """
        gen = read_excel(file_path, chunksize=batch_size, iterator=True, get_buffer=False, **kwargs)
        for batch in gen:
            yield batch

    @staticmethod
    def _from_excel(file_path: FilePathOrBuffer, batch_size=Union[None, int], **kwargs) -> Any:
        """
        Method to import the dataframe from excel.
        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the dataframe will be imported.

        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        if not batch_size:

            df = read_excel(file_path, **kwargs)
            return df

        else:
            return ExcelMixin.retrieve_a_generator(file_path, batch_size, **kwargs)

    @classmethod
    def from_excel(cls, file_path: FilePathOrBuffer, **kwargs) -> 'ExcelMixin':
        """
        Method to import the dataframe from excel.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the dataframe will be imported.

        Returns
        -------
        ExcelMixin: object of the class that inherits from ExcelMixin

        """
