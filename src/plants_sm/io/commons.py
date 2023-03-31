import io
import tarfile
from pathlib import Path
from typing import IO, TextIO, Union, AnyStr
from urllib.parse import urlparse

import requests

FilePathOrBuffer = Union[str, Path, IO[AnyStr], TextIO]
Buffer = Union[TextIO, IO[AnyStr]]


def get_handle_to_write(filepath_or_buffer: FilePathOrBuffer, mode: str = 'r', **kwargs) -> IO[AnyStr]:
    """
    Function that returns the file handler from the Path object or the buffer object or the url.

    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path

    mode : str, optional
        file mode, by default 'r'

    Returns
    -------
    handler : IO[AnyStr]
        file handler

    Raises
    ------
    ValueError
        if the filepath_or_buffer is not a valid file path or buffer or url
    """

    if is_buffer(filepath_or_buffer):
        return filepath_or_buffer

    elif is_url(filepath_or_buffer, **kwargs):
        raise ValueError("Cannot write to a url.")

    return open(filepath_or_buffer, mode=mode)


def is_url(filepath_or_buffer: FilePathOrBuffer, **kwargs) -> bool:
    """
    Function that checks if the filepath_or_buffer is a url.

    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path

    Returns
    -------
    is_url : bool
        True if the filepath_or_buffer is an url, False otherwise.
    """
    if not isinstance(filepath_or_buffer, str):
        return False

    if urlparse(filepath_or_buffer).scheme != '':
        try:

            response = requests.head(filepath_or_buffer, allow_redirects=True, **kwargs)
            if response.status_code == 200:
                return True

            return False

        except requests.exceptions.RequestException:
            return False

    return False


def is_buffer(filepath_or_buffer: FilePathOrBuffer) -> bool:
    """
    Function that checks if the filepath_or_buffer is a buffer.

    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path

    Returns
    -------
    is_buffer : bool
        True if the filepath_or_buffer is a buffer, False otherwise.
    """
    return isinstance(filepath_or_buffer, (io.TextIOBase,
                                           io.BufferedIOBase,
                                           io.RawIOBase,
                                           io.IOBase))


def is_path(filepath_or_buffer: FilePathOrBuffer) -> bool:
    """
    Function that checks if the filepath_or_buffer is a path.

    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path

    Returns
    -------
    is_path : bool
        True if the filepath_or_buffer is a path, False otherwise.
    """
    if isinstance(filepath_or_buffer, Path):
        return True

    path = Path(filepath_or_buffer)
    try:
        if path.exists():
            return True
        return False

    except OSError:
        return False


def is_dir(filepath_or_buffer: FilePathOrBuffer) -> bool:
    """
    Function that checks if the DirPath is a directory.

    Parameters
    ----------
    filepath_or_buffer : str | Path
        directory path path

    Returns
    -------
    is_path : booldir_path
        True if the dir_path is a directory, False otherwise.
    """
    if not isinstance(filepath_or_buffer, Path):
        filepath_or_buffer = Path(filepath_or_buffer)
    try:
        if filepath_or_buffer.is_dir():
            return True
        return False

    except OSError:
        return False


def is_gzip_tarball(filepath_or_buffer: FilePathOrBuffer) -> bool:
    """
    Function that checks if the FilePathOrBuffer is a gziped tarball.

    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        directory path path

    Returns
    -------
    is_path : bool
        True if the FilePathOrBuffer is a gziped tarball, False otherwise.
    """
    try:
        if tarfile.is_tarfile(filepath_or_buffer):
            return True
        return False

    except OSError:
        return False


def is_gzip_tarball_path(filepath_or_buffer: FilePathOrBuffer) -> bool:
    """
    Checks that the provided path is in good-faith a .tar.gz tarball.

    It only checks the path for situations where it is not possible to check the content directly, like in URL before
    downloading. This is a very unsafe assumption! As soon as you can, you must use the is_gzip_tarball() function to
    check the contents directly.

    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        directory path path

    Returns
    -------
    is_path : bool
        True if the FilePathOrBuffer is a gziped tarball, False otherwise.
    """
    suffix = ".tar.gz"

    if isinstance(filepath_or_buffer, Path):
        if filepath_or_buffer.name.endswith(suffix):
            return True
        return False
    try:
        if filepath_or_buffer.endswith(suffix):
            return True
        return False

    except OSError:
        return False


def get_handle_to_read(filepath_or_buffer: FilePathOrBuffer, mode: str = 'r', **kwargs) -> IO[AnyStr]:
    """
    Function that returns the file handler from the Path object or the buffer object or the url.

    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[AnyStr] | TextIO
        file path

    mode : str, optional
        file mode, by default 'r'

    Returns
    -------
    handler : IO[AnyStr]
        file handler

    Raises
    ------
    ValueError
        if the filepath_or_buffer is not a valid file path or buffer or url
    """
    if is_buffer(filepath_or_buffer):
        return filepath_or_buffer

    elif is_path(filepath_or_buffer):
        return open(filepath_or_buffer, mode=mode)

    elif is_url(filepath_or_buffer, **kwargs):
        data = requests.get(filepath_or_buffer, **kwargs)

        if is_gzip_tarball_path(filepath_or_buffer):
            tar = tarfile.open(fileobj=io.BytesIO(data.content))
            return tar.extractfile(tar.getnames()[0])

        if mode == 'r':
            return io.StringIO(data.text)

        elif mode == 'rb':
            return io.BytesIO(data.content)

        raise ValueError("Cannot read from a url.")

    raise ValueError("No file path or buffer or url provided.")


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