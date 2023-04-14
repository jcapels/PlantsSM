from abc import ABCMeta, abstractmethod
from typing import Any, List

from plants_sm.io.commons import FilePathOrBuffer, get_buffer


class Reader(metaclass=ABCMeta):
    """
    Abstract class that aims at reading files of any format
    """

    def __init__(self, file_path_or_buffer: FilePathOrBuffer, **kwargs):
        """
        Constructor

        Parameters
        ----------
        file_path_or_buffer: FilePathOrBuffer

        """
        self.path = file_path_or_buffer
        if "get_buffer" in kwargs:
            del kwargs["get_buffer"]
        else:
            if "mode" in kwargs:
                self.buffer = get_buffer(file_path_or_buffer, mode=kwargs["mode"])
                del kwargs["mode"]
            else:
                self.buffer = get_buffer(file_path_or_buffer)
        self.kwargs = kwargs

    def close_buffer(self):
        """
        Method to close buffer.
        """
        self.buffer.close()

    @staticmethod
    @abstractmethod
    def file_types() -> List[str]:
        """
        Abstract method and property that returns the file types that the reader can read.

        Returns
        -------
        file_types : List[str]
            the file types that the reader can read.
        """

    @abstractmethod
    def read(self) -> Any:
        """
        Abstract method that aims at reading a file.

        Returns
        -------
        Any object with the information contained in the file.
        """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_buffer()
        return False
