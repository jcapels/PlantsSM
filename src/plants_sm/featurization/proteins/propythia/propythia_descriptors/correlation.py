from collections import Counter
from statistics import stdev

from modlamp.descriptors import PeptideDescriptor

from ._constants import (COR_HYDROPHOBICITY_NORMALIZED,
                         COR_AV_FLEXIBILITY_NORMALIZED,
                         COR_POLARIZABILITY_NORMALIZED,
                         COR_FREE_ENERGY_NORMALIZED,
                         COR_RESIDUE_ASA_NORMALIZED,
                         COR_RESIDUE_VOL_NORMALIZED,
                         COR_STERIC_NORMALIZED,
                         COR_MUTABILITY_NORMALIZED)
from ._utils import Descriptor

_AAPS = [COR_HYDROPHOBICITY_NORMALIZED,
         COR_AV_FLEXIBILITY_NORMALIZED,
         COR_POLARIZABILITY_NORMALIZED,
         COR_FREE_ENERGY_NORMALIZED,
         COR_RESIDUE_ASA_NORMALIZED,
         COR_RESIDUE_VOL_NORMALIZED,
         COR_STERIC_NORMALIZED,
         COR_MUTABILITY_NORMALIZED]


_AAPS_NAMES = ['COR_HYDROPHOBICITY', 'COR_AV_FLEXIBILITY', 'COR_POLARIZABILITY', 'COR_FREE_ENERGY', 'COR_RESIDUE_ASA',
               'COR_RESIDUE_VOL', 'COR_STERIC', 'COR_MUTABILITY']


def calculate_each_normalized_moreau_broto_auto(protein_sequence, aap):
    result = []
    for i in range(1, 31):
        temp = 0

        for j in range(len(protein_sequence) - i):
            temp += aap[protein_sequence[j]] * aap[protein_sequence[j + 1]]

        if len(protein_sequence) - i == 0:
            result.append(round(temp / (len(protein_sequence)), 3))

        else:
            result.append(round(temp / (len(protein_sequence) - i), 3))

    return result


def calculate_each_moran_auto(protein_sequence, aap):
    counts = Counter(protein_sequence)
    cds = sum([cnt * aap[aa] for aa, cnt in counts.items()])
    pmean = cds / len(protein_sequence)

    cc = [aap[aa] for aa in protein_sequence]
    k = (stdev(cc)) ** 2

    result = []
    for i in range(1, 31):
        temp = 0

        for j in range(len(protein_sequence) - i):
            temp += (aap[protein_sequence[j]] - pmean) * (aap[protein_sequence[j + i]] - pmean)

        if len(protein_sequence) - i == 0:
            result.append(round(temp / (len(protein_sequence)) / k, 3))

        else:
            result.append(round(temp / (len(protein_sequence) - i) / k, 3))

    return result


def calculate_each_geary_auto(protein_sequence, aap):

    cc = [aap[aa] for aa in protein_sequence]

    k = (((stdev(cc)) ** 2) * len(protein_sequence)) / (len(protein_sequence) - 1)

    result = []
    for i in range(1, 31):
        temp = 0

        for j in range(len(protein_sequence) - i):
            temp += (aap[protein_sequence[j]] - aap[protein_sequence[j + i]]) ** 2

        if len(protein_sequence) - i == 0:
            result.append(round(temp / (2 * (len(protein_sequence))) / k, 3))

        else:
            result.append(round(temp / (2 * (len(protein_sequence) - i)) / k, 3))

    return result


class NormalizedMoreauBrotoDescriptor(Descriptor):
    """
    A descriptor that returns the normalized Moreau Broto of the sequence.
    """

    def __call__(self, sequence: str, lambda_: int = 10, weight: float = 0.05, **kwargs):
        """
        Calculates the normalized Moreau Broto of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the normalized Moreau Broto for.

        lambda_ : int, default=10
            The lambda value to use for the calculation.

        weight : float, default=0.05
            The weight to use for the calculation.

        kwargs

        Returns
        -------
        list of float
            The normalized Moreau Broto of the sequence.
        """
        result = []
        for aap in _AAPS:
            result.extend(calculate_each_normalized_moreau_broto_auto(sequence, aap))
        return result

    def get_features_out(self, **kwargs):
        return [f'MoreauBrotoAuto_{aap}_{i}' for aap in _AAPS_NAMES for i in range(1, 31)]


class MoranAutoDescriptor(Descriptor):
    """
    A descriptor that returns the Moran Auto of the sequence.
    """

    def __call__(self, sequence: str, lambda_: int = 10, weight: float = 0.05, **kwargs):
        """
        Calculates the Moran Auto of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the Moran Auto for.

        lambda_ : int, default=10
            The lambda value to use for the calculation.

        weight : float, default=0.05
            The weight to use for the calculation.

        kwargs

        Returns
        -------
        list of float
            The Moran Auto of the sequence.
        """
        result = []
        for aap in _AAPS:
            result.extend(calculate_each_moran_auto(sequence, aap))
        return result

    def get_features_out(self, **kwargs):
        return [f'MoranAuto_{aap}_{i}' for aap in _AAPS_NAMES for i in range(1, 31)]


class GearyAutoDescriptor(Descriptor):
    """
    A descriptor that returns the Geary Auto of the sequence.
    """

    def __call__(self, sequence: str, lambda_: int = 10, weight: float = 0.05, **kwargs):
        """
        Calculates the Geary Auto of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the Geary Auto for.

        lambda_ : int, default=10
            The lambda value to use for the calculation.

        weight : float, default=0.05
            The weight to use for the calculation.

        kwargs

        Returns
        -------
        list of float
            The Geary Auto of the sequence.
        """
        result = []
        for aap in _AAPS:
            result.extend(calculate_each_geary_auto(sequence, aap))
        return result

    def get_features_out(self, **kwargs):
        return [f'GearyAuto_{aap}_{i}' for aap in _AAPS_NAMES for i in range(1, 31)]


class ModlAMPAutoCorrelationDescriptor(Descriptor):
    """
    A descriptor that returns the Auto correlation of the sequence inferred with modlAMP.
    """
    def __call__(self, sequence: str, window: int = 7, scalename: str = 'Eisenberg', **kwargs):
        """
        Calculates the Auto correlation of the sequence inferred with modlAMP.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the Auto correlation inferred with modlAMP for.

        window : int, default=7
            The window to use for the calculation.

        scalename : str, default='Eisenberg'
            The scalename to use for the calculation.

        kwargs

        Returns
        -------
        list of float
            The the Auto correlation of the sequence inferred with modlAMP.
        """
        amp = PeptideDescriptor(sequence, scalename)
        amp.calculate_autocorr(window)
        return list(amp.descriptor[0])

    def get_features_out(self, window: int = 7, **kwargs):
        # Auto correlation yields window n values for Eisenberg scalename
        return [f'modlAMP_auto_correlation_{i}' for i in range(1, window + 1)]


class ModlAMPCrossCorrelationDescriptor(Descriptor):
    """
    A descriptor that returns the Cross correlation of the sequence inferred with modlAMP.
    """
    def __call__(self, sequence: str, window: int = 7, scalename: str = 'Eisenberg', **kwargs):
        """
        Calculates the Cross correlation of the sequence inferred with modlAMP.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the Cross correlation inferred with modlAMP for.

        window : int, default=7
            The window to use for the calculation.

        scalename : str, default='Eisenberg'
            The scalename to use for the calculation.

        kwargs

        Returns
        -------
        list of float
            The the Auto correlation of the sequence inferred with modlAMP.
        """
        amp = PeptideDescriptor(sequence, scalename)
        amp.calculate_crosscorr(window)
        return list(amp.descriptor[0])

    def get_features_out(self, window: int = 7, **kwargs):
        # Cross correlation yields window n values for Eisenberg scalename
        return [f'modlAMP_cross_correlation_{i}' for i in range(1, window + 1)]
