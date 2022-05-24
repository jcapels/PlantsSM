from typing import List, Any, Union

import pandas as pd

from plants_sm.data_structures.dataset import PandasDataset
from plants_sm.io.commons import FilePathOrBuffer
from plants_sm.io.reader import Reader
from plants_sm.io.writer import Writer


class ExcelReader(Reader):

    def __init__(self, filepath_or_buffer: FilePathOrBuffer, **kwargs):
        """
        Initializer of this class that defines instance variables by calling the initializer of the parent class (
        Reader).

        Parameters
        ----------
        filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
            file path
        sep : string
            the separator of each column in the file

        Returns
        -------
        """
        super().__init__(filepath_or_buffer, mode="rb", **kwargs)

    @property
    def file_types(self) -> List[str]:
        """
        File types that the reader can read

        Returns
        -------
        file types: List[str]
            List of file types that the reader can read
        """
        return ["xlsx"]

    def read(self) -> pd.DataFrame:
        """
        Method to read the inputted file or buffer

        Returns
        -------
        dataframe: pd.DataFrame
            dataframe with the read data
        """
        return pd.read_excel(self.buffer, **self.kwargs)


class ExcelWriter(Writer):

    def __init__(self, filepath_or_buffer: FilePathOrBuffer, **kwargs):
        """
        Initializer of this class that defines instance variables by calling the initializer of the parent class (
        Writer) and the pd.DataFrame to be written.

        Parameters
        ----------
        filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
            file path
        sep : string
            the separator of each column in the file

        Returns
        -------
        """

        super().__init__(filepath_or_buffer, **kwargs)

    @property
    def file_types(self) -> List[str]:
        """
        File types that the reader can read

        Returns
        -------
        file types: List[str]
            List of file types that the reader can read
        """
        return ["xlsx", "xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]

    def write(self, object_to_be_written: pd.DataFrame) -> bool:
        """

        Parameters
        ----------
        object_to_be_written: pd.DataFrame
            pandas DataFrame to be written

        Returns
        -------
        flag : boolean
            whether the pd.DataFrame was written without errors or not
        """
        try:
            object_to_be_written.to_excel(self.path, **self.kwargs)
            return True
        except FileNotFoundError:
            return False


def read_excel(file_path: FilePathOrBuffer, **kwargs) -> pd.DataFrame:
    """
    Function to read Excel file calling the ExcelReader method.

    Parameters
    ----------
    file_path: FilePathOrBuffer
        path of the file to be read
    Returns
    -------
    data: pd.DataFrame
        pandas dataframe with the read data
    """
    reader = ExcelReader(file_path, **kwargs)
    df = reader.read()
    reader.close_buffer()
    return df


def write_excel(output_path: FilePathOrBuffer, object_to_be_written: pd.DataFrame, **kwargs) -> bool:
    """
    Function to write an Excel file calling the ExcelWriter method.

    Parameters
    ----------
    output_path: FilePathOrBuffer
        path or buffer of the file to be written
    object_to_be_written: pd.DataFrame
        pandas dataframe to be written

    Returns
    -------
    flag:bool
        flag that says whether the file was written
    """
    writer = ExcelWriter(filepath_or_buffer=output_path, **kwargs)
    correctly_written = writer.write(object_to_be_written)
    writer.close_buffer()
    return correctly_written


# TODO: add a Mixin to cope with this part
def excel_to_dataset(file_path: FilePathOrBuffer,
                     representation_field: Union[str, List[Union[str, int]]] = None,
                     features_field: Union[str, List[Union[str, int]]] = None,
                     labels_field: Union[str, List[Union[str, int]]] = None,
                     instances_ids_field: Union[str, List[Union[str, int]]] = None,
                     **kwargs) -> PandasDataset:
    """
    Read an excel file

    Parameters
    ----------
    file_path: FilePathOrBuffer
        path of the file to be read
    representation_field: str | List[str | int] (optional)
        representation column field (to be processed)
    features_field: str | List[str | int] (optional)
        features column field
    labels_field: str | List[str | int] (optional)
        labels column field
    instances_ids_field: str | List[str | int] (optional)
        instances column field

    Returns
    ----------
    dataset: PandasDataset
        dataset with information for ML tasks
    """

    reader = ExcelReader(file_path, **kwargs)
    df = reader.read()
    reader.close_buffer()
    dataset = PandasDataset(df, representation_field, features_field, labels_field, instances_ids_field)
    return dataset
