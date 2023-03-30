from pathlib import Path
from typing import IO, TextIO, Union, AnyStr
from contextlib import contextmanager

FilePathOrBuffer = Union[str, Path, IO[AnyStr], TextIO]
Buffer = Union[TextIO, IO[AnyStr]]


@contextmanager
def buffer(filepath_or_buffer: FilePathOrBuffer, mode: str = 'r', **kwargs) -> Buffer:
    """
    Function that uses the context manager annotator and yields the buffer
    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path
    mode : str
        mode of buffering (e.g., 'w' for writing, 'r' for reading)
    Returns
    -------
    """
    if hasattr(filepath_or_buffer, 'buffer'):
        buf = filepath_or_buffer

    else:
        buf = open(filepath_or_buffer, mode, **kwargs)

    try:

        yield buf

    finally:
        if isinstance(buf, TextIO) or isinstance(buf, IO):
            buf.close()


def get_buffer(filepath_or_buffer: FilePathOrBuffer, mode: str = 'r', **kwargs) -> Buffer:
    """
    Function that opens the file and returns a Buffer object
    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path
    mode : str
        mode of buffering (e.g., 'w' for writing, 'r' for reading)
    Returns
    -------
    """
    if hasattr(filepath_or_buffer, 'buffer'):
        return filepath_or_buffer

    else:
        return open(filepath_or_buffer, mode, **kwargs)


def get_path(filepath_or_buffer: FilePathOrBuffer, **kwargs) -> Path:
    """
    Function that returns a Path object.
    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path
    Returns
    -------
    path : Path
    """
    if isinstance(filepath_or_buffer, TextIO) or isinstance(filepath_or_buffer, IO):
        return Path(filepath_or_buffer.name, **kwargs)

    return Path(filepath_or_buffer, **kwargs)
