from abc import abstractmethod
from itertools import tee
from typing import List, Iterable

from joblib import Parallel, delayed
from statistics import mean, stdev


class Descriptor:
    """
    A descriptor is a function that takes a list of atoms and returns a list of floats.
    """

    @abstractmethod
    def __call__(self, sequence: str, **kwargs):
        """
        It calculates the descriptor for a given sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the descriptor for.

        kwargs
            Keyword arguments that are passed to the descriptor.

        Returns
        -------
        list of float
            A list of floats.
        """
        pass

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def get_features_out(self, **kwargs):
        """
        Returns the features that this descriptor will return.

        Parameters
        ----------
        kwargs
            Keyword arguments that are passed to the descriptor.

        Returns
        -------
        list
            A list of strings.
        """
        pass


def calculate_descriptor(descriptor: Descriptor, sequences: List[str], n_jobs: int = 1, **kwargs):
    """
    It calculates the descriptor for a list of sequences in parallel.

    Parameters
    ----------
    descriptor : Descriptor
        The descriptor to make parallel.

    sequences : list of str
        The sequences to calculate the descriptor for.

    n_jobs : int
        The number of jobs to run in parallel.

    kwargs
        Keyword arguments that are passed to the descriptor.

    Returns
    -------
    list of list of float
        The list of the descriptor for each sequence.
    """
    parallel = Parallel(n_jobs=n_jobs)
    return parallel(delayed(descriptor)(sequence, **kwargs) for sequence in sequences)


def normalize_aap(aap: dict) -> dict:
    """
    Normalize each amino acid property.

    Parameters
    ----------
    aap : dict
        The amino acid property.

    Returns
    -------
    aap_norm : dict
        The normalized amino acid property.
    """
    _mean = mean(list(aap.values()))
    _stdev = stdev(list(aap.values()))

    result = {aa: (j - _mean) / _stdev for aa, j in aap.items()}
    return result


def pairwise(iterable: Iterable, n: int = 1) -> Iterable:
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ..., n = 1
    s -> (s0,s2), (s1,s3), (s2, s4), ..., n = 2

    Parameters
    ----------
    iterable : iterable
        The iterable to pairwise.

    n : int, optional (default = 1)
        The number of elements to pairwise.

    Returns
    -------
    list of tuples iterator : Iterable
        The list of tuples.
    """
    a, b = tee(iterable)
    for i in range(n):
        next(b, None)
    return zip(a, b)
