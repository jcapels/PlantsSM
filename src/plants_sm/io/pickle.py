import pickle
from typing import List, Any

from plants_sm.io.commons import FilePathOrBuffer
from plants_sm.io.reader import Reader
from plants_sm.io.writer import Writer


class PickleReader(Reader):

    def __init__(self, filepath_or_buffer: FilePathOrBuffer, **kwargs):
        """
        Initializer of this class that defines instance variables by calling the initializer of the parent class (
        Reader).

        Parameters
        ----------
        filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
            file path
        """
        super().__init__(filepath_or_buffer, mode="rb", **kwargs)

    @staticmethod
    def file_types() -> List[str]:
        """
        Returns the file types that the pickle reader can read.

        Returns
        -------
        file_types : List[str]
            object that returns the file types that the pickle reader can read.
        """
        return ['pkl']

    def read(self) -> Any:
        """
        Method to read the data from file(s) and returns a pandas pd.DataFrame.

        Returns
        -------

        """
        return pickle.load(self.buffer)


class PickleWriter(Writer):

    def __init__(self, filepath_or_buffer: FilePathOrBuffer, **kwargs):
        """
        Initializer of this class that defines instance variables by calling the initializer of the parent class (
        Reader).

        Parameters
        ----------
        filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
            file path
        """
        super().__init__(filepath_or_buffer, mode="wb", **kwargs)

    @staticmethod
    def file_types() -> List[str]:
        """
        Returns the file types that the pickle writer can read.

        Returns
        -------
        file_types : List[str]
            object that returns the file types that the pickle writer can read.
        """
        return ['pkl']

    def write(self, object_to_be_written: Any) -> bool:
        """
        Method to write the pd.DataFrame into a pickle file.

        Parameters
        ----------
        object_to_be_written: Any
            object to be written

        Returns
        -------
        bool
            True if the file was written successfully, False otherwise.
        """
        pickle.dump(object_to_be_written, self.buffer)
        return True


def read_pickle(path: str) -> Any:
    """
    Reads a pickle file and returns the object.

    Parameters
    ----------
    path: str
        The path to the pickle file.

    Returns
    -------
    Any
        The object that was saved in the pickle file.
    """
    return PickleReader(path).read()


def write_pickle(path: str, object_to_be_written: Any) -> bool:
    """
    Writes an object into a pickle file.

    Parameters
    ----------
    path: str
        The path to the pickle file.
    object_to_be_written: Any
        The object to be written.

    Returns
    -------
    bool
        True if the file was written successfully, False otherwise.
    """
    return PickleWriter(path).write(object_to_be_written)


def is_pickable(object_to_be_written: Any):
    """
    Checks if an object is pickable.

    Parameters
    ----------
    object_to_be_written: Any
        The object to be written.

    Returns
    -------

    """
    try:
        pickle.dumps(object_to_be_written)
        return True
    except Exception:
        return False
