from typing import List

import pandas as pd
from plants_sm.io.commons import FilePathOrBuffer
from plants_sm.io.reader import Reader
from plants_sm.io.writer import Writer


class CSVReader(Reader):
    """
    Class that implements the class Reader and reads CSV, TSV and other format files.
    """

    def __init__(self, filepath_or_buffer: FilePathOrBuffer, sep: str = ",", **kwargs):
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
        super().__init__(filepath_or_buffer, sep=sep, **kwargs)

    @staticmethod
    def file_types() -> List[str]:
        """
        Returns the file types that the reader can read.

        Returns
        -------
        file_types : List[str]
            object that returns the file types that the reader can read.
        """
        return ['txt', 'csv', 'tsv']

    def read(self) -> pd.DataFrame:
        """
        Method to read the data from file(s) and returns a pandas pd.DataFrame.

        Parameters
        ----------

        Returns
        -------
        data : pd.DataFrame
            Read DataFrame
        """
        return pd.read_csv(self.path, **self.kwargs)


class CSVWriter(Writer):
    """
    Class that implements the class Writer and writes CSV, TSV and other format files.
    """

    def __init__(self, filepath_or_buffer: FilePathOrBuffer, sep: str = ",", **kwargs):
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

        super().__init__(filepath_or_buffer, sep=sep, **kwargs)

    @staticmethod
    def file_types() -> List[str]:
        """
        Returns the file types that the writer can write.

        Returns
        -------
        file_types : List[str]
            the file types that the writer can write.
        """
        return ['txt', 'csv', 'tsv']

    def write(self, df: pd.DataFrame) -> bool:
        """
        Method to write the pd.DataFrame into a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            pandas DataFrame to be written

        Returns
        -------
        flag : boolean
            whether the pd.DataFrame was written without errors or not
        """
        try:
            df.to_csv(self.buffer, **self.kwargs)
            return True
        except FileNotFoundError:
            return False


def read_csv(filepath_or_buffer: FilePathOrBuffer, **kwargs) -> pd.DataFrame:
    """
    Function that reads the CSV file and returns a pandas pd.DataFrame.

    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path

    Returns
    -------
    df : pd.DataFrame
        Read DataFrame
    """

    reader = CSVReader(filepath_or_buffer=filepath_or_buffer, **kwargs)
    df = reader.read()
    return df


def write_csv(filepath_or_buffer: FilePathOrBuffer, df: pd.DataFrame, **kwargs) -> bool:
    """
    Function that writes a pandas pd.DataFrame in a CSV file.

    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path

    df : pd.DataFrame
        Dataframe to be written in the CSV file

    Returns
    -------
    flag : boolean
        whether the pd.DataFrame was written without errors
    """

    writer = CSVWriter(filepath_or_buffer=filepath_or_buffer, **kwargs)
    ddf = writer.write(df=df)
    writer.close_buffer()
    return ddf
