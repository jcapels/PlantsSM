from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import List, Any

from plants_sm.io.commons import FilePathOrBuffer, get_path, get_buffer


class Writer(metaclass=ABCMeta):
    """
    Generic class for writing data from file(s).
    All writers must implement the read method.
    All writers must define the file_types attribute.
    """

    def __init__(self, filepath_or_buffer: FilePathOrBuffer, **kwargs):
        """
        Initializer of this class that defines instance variables such as the buffer for the file and path.

        Parameters
        ----------
        filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
            file path

        Returns
        -------
        """

        self.path = get_path(filepath_or_buffer)
        self.buffer = get_buffer(filepath_or_buffer, mode="w")
        self.kwargs = kwargs

    def close_buffer(self):
        """
        Method to close buffer.
        """
        self.buffer.close()

    @property
    @abstractmethod
    def file_types(self) -> List[str]:
        """
        Abstract method and property that returns the file types that the writer can write.

        Returns
        -------
        file_types : List[str]
            file types that the writer can write
        """

    @contextmanager
    @abstractmethod
    def write(self, object_to_be_written: Any) -> bool:
        """
        Abstract method to read the data from file(s).

        Parameters
        ----------
        object_to_be_written : Any
            object to be written (e.g., dataframe or configuration file)

        Returns
        -------
        bool : True if the data was written successfully, False otherwise.
        """
        pass
