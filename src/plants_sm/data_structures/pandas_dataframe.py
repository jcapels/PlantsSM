from types import ModuleType
from typing import Optional

df_libs = {}

try:
    import cudf as cf
    df_libs['cudf'] = cf
except ImportError:
    cf = None

try:
    import modin.pandas as mod
    df_libs['modin'] = mod
except ImportError:
    mod = None

try:
    import pandas as pd
    df_libs['pandas'] = pd
except ImportError:
    pd = None


default_df_lib = None


def get_default_df() -> Optional[str]:
    """
    It returns the name of the default dataframe library.

    Returns
    -------
    df_lib: str
        The default dataframe library name (currently available: 'cudf', 'modin', 'pandas')
    """
    global default_df_lib
    df_lib_order = ['cudf', 'modin', 'pandas']

    if not default_df_lib:
        for df_lib in df_lib_order:
            if df_lib in df_libs:
                default_df_lib = df_lib
                break

        if not default_df_lib:
            raise RuntimeError("No dataframe library available.")

    return default_df_lib


def set_default_df(df_lib: str):
    """
    It sets the default dataframe library.

    Parameters
    ----------
    df_lib: str
        The dataframe library name (currently available: 'cudf', 'modin', 'pandas')

    Returns
    -------
    """

    global default_df_lib

    df_lib = df_lib.lower()

    if df_lib in df_libs:
        default_df_lib = df_lib
    else:
        raise RuntimeError(f"dataframe library {df_lib} not available.")


def df_instance() -> ModuleType:
    """
    It returns the selected dataframe library.

    Returns
    -------
    df: ModuleType
        The default dataframe library instance (currently available: 'cudf', 'modin', 'pandas')
    """

    df_name = get_default_df()

    if df_name:
        return df_libs[df_name]


pd = df_instance()
