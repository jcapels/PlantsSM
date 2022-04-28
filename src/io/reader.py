from abc import ABCMeta, abstractmethod
from typing import Any

from io.commons import FilePathOrBuffer, get_path, get_buffer


class Reader(metaclass=ABCMeta):
    """
    Abstract class that aims at reading files of any format
    """

    def __init__(self, file_path_or_buffer: FilePathOrBuffer, **kwargs):
        """

        Parameters
        ----------
        file_path_or_buffer
        """
        self.path = get_path(file_path_or_buffer)
        self.buffer = get_buffer(file_path_or_buffer)
        self.kwargs = kwargs

    @abstractmethod
    def read(self) -> Any:
        """

        Returns
        -------

        """
        pass
