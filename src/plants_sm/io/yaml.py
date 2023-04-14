from typing import Any, Dict, List, Union, IO, TextIO

import dask
from yaml import CLoader as Loader, safe_load
from yaml import load, dump

from plants_sm.io.commons import get_handle_to_read, get_handle_to_write, FilePathOrBuffer
from plants_sm.io.reader import Reader
from plants_sm.io.writer import Writer


class YAMLReader(Reader):

    @staticmethod
    def file_types() -> List[str]:
        """
        Returns the file types that the reader can read.

        Returns
        -------
        file_types : List[str]
            file types that the reader can read.
        """
        return ['yml', 'yaml']

    @dask.delayed
    def read(self) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Method to read the data from file(s) and returns a dictionary or list of dictionaries.

        Parameters
        ----------

        Returns
        -------
        data : Union[List[Dict[str, Any]], Dict[str, Any]]
        """
        if 'Loader' not in self.kwargs:
            self.kwargs['Loader'] = Loader

        mode = self.kwargs.pop('mode', 'r')
        auth = self.kwargs.pop('auth', None)
        if "preserve_order" in self.kwargs:
            preserve_order = self.kwargs.pop("preserve_order")
        else:
            preserve_order = False
        handle = get_handle_to_read(self.buffer, mode=mode, auth=auth)

        with handle as f:
            if preserve_order:
                return load(f, **self.kwargs)
            else:
                return safe_load(f)


class YAMLWriter(Writer):

    @staticmethod
    def file_types() -> List[str]:
        """
        Returns the file types that the writer can write.

        Returns
        -------
        file_types : List[str]
            file types that the writer can write.
        """
        return ['yml', 'yaml']

    def write(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> bool:
        """
        Method to write the dictionary into a YAML file.

        Parameters
        ----------
        data : Any
            object to be written (e.g., dataframe or data file)

        Returns
        -------
        flag : boolean
            whether the DataFrame was written without errors or not
        """
        mode = self.kwargs.pop('mode', 'w')
        handle = get_handle_to_write(self.buffer, mode=mode)
        with handle as f:
            dump(data, f, **self.kwargs)
        return True


def read_yaml(file_path_or_buffer: FilePathOrBuffer, **kwargs) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Function that reads the YAML file and returns a dictionary.

    Parameters
    ----------
    file_path_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path

    Returns
    -------
    data : Union[List[Dict[str, Any]], Dict[str, Any]]
        dictionary with the information read from the YAML file
    """
    with YAMLReader(file_path_or_buffer=file_path_or_buffer, **kwargs) as reader:
        # suppress warning due to dask delayed decorator: https://github.com/dask/dask/issues/7779
        # noinspection PyUnresolvedReferences
        data = reader.read().compute()
    return data


def write_yaml(file_path_or_buffer: FilePathOrBuffer,
               data: Union[List[Dict[str, Any]], Dict[str, Any]],
               **kwargs) -> bool:
    """
    Function that writes a dictionary in a YAMl file.

    Parameters
    ----------
    file_path_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path
    data : Union[List[Dict[str, Any]], Dict[str, Any]]
        object with the information to be written

    Returns
    -------
    flag : boolean
        whether the DataFrame was written without errors
    """
    with YAMLWriter(file_path_or_buffer=file_path_or_buffer, **kwargs) as writer:
        flag = writer.write(data=data)
    return flag
