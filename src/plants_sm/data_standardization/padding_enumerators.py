from enum import Enum


class PaddingEnumerators(Enum):
    """
    Enumerators of characters used to pad sequences.
    """

    COMPOUNDS = "G"
    PROTEINS = "-"
