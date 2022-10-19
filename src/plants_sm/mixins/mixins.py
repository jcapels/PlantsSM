import os.path
import re
from abc import abstractmethod
from typing import Any

import pandas as pd

from plants_sm.io import read_csv, write_csv
from plants_sm.io.commons import FilePathOrBuffer
from plants_sm.io.excel import write_excel, read_excel


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
    def _from_csv(file_path: FilePathOrBuffer, **kwargs) -> Any:
        """
        Method to import the dataframe from csv.

        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the dataframe will be imported.

        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        df = read_csv(file_path, **kwargs)

        return df

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
    def _from_excel(file_path: FilePathOrBuffer, **kwargs) -> Any:
        """
        Method to import the dataframe from excel.
        Parameters
        ----------
        file_path: FilePathOrBuffer
            path to the file where the dataframe will be imported.

        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        df = read_excel(file_path, **kwargs)

        return df

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