from typing import List, Union
import json
from plants_sm.io.commons import FilePathOrBuffer
from plants_sm.io.reader import Reader
from plants_sm.io.writer import Writer


class JSONReader(Reader):

    def __init__(self, filepath_or_buffer: FilePathOrBuffer, **kwargs):
        """
        Initializer of this class that defines instance variables by calling the initializer of the parent class (
        Reader).

        Parameters
        ----------
        filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
            file path

        Returns
        -------
        """
        super().__init__(filepath_or_buffer, mode="r", **kwargs)

    @staticmethod
    def file_types() -> List[str]:
        """
        File types that the reader can read

        Returns
        -------
        file types: List[str]
            List of file types that the reader can read
        """
        return ["json"]

    def read(self) -> Union[dict, List[dict]]:
        """
        Method to read the inputted file or buffer

        Returns
        -------
        data: Union[dict, List[dict]]
            Python object representing the JSON data
        """
        return json.load(self.buffer)


class JSONWriter(Writer):

    def __init__(self, filepath_or_buffer: FilePathOrBuffer, **kwargs):
        """
        Initializer of this class that defines instance variables by calling the initializer of the parent class (
        Writer) and the Python object to be written.

        Parameters
        ----------
        filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
            file path

        Returns
        -------
        """

        super().__init__(filepath_or_buffer, **kwargs)

    @staticmethod
    def file_types() -> List[str]:
        """
        File types that the reader can read

        Returns
        -------
        file types: List[str]
            List of file types that the reader can read
        """
        return ["json"]

    def write(self, object_to_be_written: Union[dict, List[dict]]) -> bool:
        """

        Parameters
        ----------
        object_to_be_written: Union[dict, List[dict]]
            Python object representing the JSON data to be written

        Returns
        -------
        flag : boolean
            whether the JSON data was written without errors or not
        """
        try:
            json.dump(object_to_be_written, self.buffer, **self.kwargs)
            return True
        except FileNotFoundError:
            return False


def read_json(file_path: FilePathOrBuffer, **kwargs) -> Union[dict, List[dict]]:
    """
    Function to read JSON file calling the JSONReader method.

    Parameters
    ----------
    file_path: FilePathOrBuffer
        path of the file to be read
    Returns
    -------
    data: Union[dict, List[dict]]
        Python object representing the JSON data
    """
    reader = JSONReader(file_path, **kwargs)
    data = reader.read()
    reader.close_buffer()
    return data


def write_json(output_path: FilePathOrBuffer, object_to_be_written: Union[dict, List[dict]], **kwargs) -> bool:
    """
    Function to write a JSON file calling the JSONWriter method.

    Parameters
    ----------
    object_to_be_written: Union[dict, List[dict]]
        Python object representing the JSON data to be written
    output_path: FilePathOrBuffer
        path or buffer of the file to be written

    Returns
    -------
    flag:bool
        flag that says whether the file was written
    """
    writer = JSONWriter(filepath_or_buffer=output_path, **kwargs)
    flag = writer.write(object_to_be_written)
    writer.close_buffer()
    return flag
