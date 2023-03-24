import os
import tempfile
from typing import List, Any
import h5py

from plants_sm.io.commons import FilePathOrBuffer
from plants_sm.io.reader import Reader
from plants_sm.io.writer import Writer


class H5Reader(Reader):

    def __init__(self, filepath_or_buffer: FilePathOrBuffer, **kwargs):
        """
        Initializer of this class that defines instance variables by calling the initializer of the parent class (
        Reader).

        Parameters
        ----------
        filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
            file path
        """
        super().__init__(filepath_or_buffer, mode="r", **kwargs)

    @staticmethod
    def file_types() -> List[str]:
        """
        Returns the file types that the HDF5 reader can read.

        Returns
        -------
        file_types : List[str]
            object that returns the file types that the HDF5 reader can read.
        """
        return ['h5', 'hdf5']

    def read(self) -> Any:
        """
        Method to read the data from file(s) and return an object.

        Returns
        -------
        object
            The object that was saved in the HDF5 file.
        """
        result_dict = {}
        with h5py.File(self.buffer.name, "r") as f:
            # read object from the HDF5 file
            for key in f.keys():
                result_dict[key] = f[key][:]

        if len(result_dict) == 1:
            return list(result_dict.values())[0]

        return result_dict


class H5Writer(Writer):

    def __init__(self, filepath_or_buffer: FilePathOrBuffer, **kwargs):
        """
        Initializer of this class that defines instance variables by calling the initializer of the parent class (
        Reader).

        Parameters
        ----------
        filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
            file path
        """
        super().__init__(filepath_or_buffer, mode="w", **kwargs)

    @staticmethod
    def file_types() -> List[str]:
        """
        Returns the file types that the HDF5 writer can read.

        Returns
        -------
        file_types : List[str]
            object that returns the file types that the HDF5 writer can read.
        """
        return ['h5', 'hdf5']

    def write(self, object_to_be_written: Any) -> bool:
        """
        Method to write the object into a HDF5 file.

        Parameters
        ----------
        object_to_be_written: Any
            The object to be written.

        Returns
        -------
        bool
            True if the file was written successfully, False otherwise.
        """
        with h5py.File(self.buffer.name, "w") as f:
            # write object to the HDF5 file
            if isinstance(object_to_be_written, dict):
                for key, value in object_to_be_written.items():
                    f.create_dataset(key, data=value)
            else:
                f.create_dataset('data', data=object_to_be_written)

        return True


def read_h5(path: str) -> Any:
    """
    Reads an HDF5 file and returns the object.

    Parameters
    ----------
    path: str
        The path to the HDF5 file.

    Returns
    -------
    Any
        The object that was saved in the HDF5 file.
    """
    return H5Reader(path).read()


def write_h5(object_to_be_written: Any, path: str) -> bool:
    """
    Writes an object into an HDF5 file.

    Parameters
    ----------
    object_to_be_written: Any
        The object to be written.
    path: str
        The path to the HDF5 file.

    Returns
    -------
    bool
        True if the file was written successfully, False otherwise.
    """
    return H5Writer(path).write(object_to_be_written)
