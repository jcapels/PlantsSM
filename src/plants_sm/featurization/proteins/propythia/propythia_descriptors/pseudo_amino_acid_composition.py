import math

from ._constants import (AA_ALPHABET,
                         HYDROPHOBICITY_NORMALIZED, HYDROPHILICITY_NORMALIZED, RESIDUE_MASS_NORMALIZED,
                         PK1_NORMALIZED, PK2_NORMALIZED, PI_NORMALIZED)
from ._utils import Descriptor
from .amino_acid_composition import AminoAcidCompositionDescriptor

_AAPS = [HYDROPHOBICITY_NORMALIZED, HYDROPHILICITY_NORMALIZED, RESIDUE_MASS_NORMALIZED,
         PK1_NORMALIZED, PK2_NORMALIZED, PI_NORMALIZED]


def get_correlation_function(ri='S', rj='D'):
    """
    Calculates the correlation function for a given pair of amino acids.

    Parameters
    ----------
    ri
    rj
    """
    theta = 0.0
    for aap in _AAPS:
        theta += math.pow(aap[ri] - aap[rj], 2)

    # normalize by the number of AAPs
    result = round(theta / 6, 3)
    return result


def get_sequence_order_correlation_factor(protein_sequence, k=1):
    """
    Calculates the sequence order correlation factor for a given protein sequence.

    Parameters
    ----------
    protein_sequence
    k
    """
    length_sequence = len(protein_sequence)
    res = []
    for i in range(length_sequence - k):
        aa1 = protein_sequence[i]
        aa2 = protein_sequence[i + k]
        res.append(get_correlation_function(aa1, aa2))
    result = round(sum(res) / (length_sequence - k), 3)
    return result


def get_correlation_function_for_apaac(ri='S', rj='D'):
    """
    Calculates the correlation function for a given pair of amino acids.

    Parameters
    ----------
    ri  : str
        The first amino acid.
    rj  : str
        The second amino acid.
    """

    theta1 = round(HYDROPHOBICITY_NORMALIZED[ri] * HYDROPHOBICITY_NORMALIZED[rj], 3)
    theta2 = round(HYDROPHILICITY_NORMALIZED[ri] * HYDROPHILICITY_NORMALIZED[rj], 3)

    return theta1, theta2


def get_sequence_order_correlation_factor_for_apaac(protein_sequence, k=1):
    """
    Calculates the sequence order correlation factor for a given protein sequence.

    Parameters
    ----------
    protein_sequence
    k
    """
    length_sequence = len(protein_sequence)
    res_hydrophobicity = []
    res_hydrophilicity = []

    for i in range(length_sequence - k):
        aa1 = protein_sequence[i]
        aa2 = protein_sequence[i + k]
        hydrophobicity, hydrophilicity = get_correlation_function_for_apaac(aa1, aa2)
        res_hydrophobicity.append(hydrophobicity)
        res_hydrophilicity.append(hydrophilicity)

    return (round(sum(res_hydrophobicity) / (length_sequence - k), 3),
            round(sum(res_hydrophilicity) / (length_sequence - k), 3))


class PseudoAminoAcidCompositionDescriptor(Descriptor):
    """
    A descriptor that returns the Pseudo Amino Acid Composition of the sequence.
    """

    def __call__(self, sequence: str, lambda_: int = 10, weight: float = 0.05, **kwargs):
        """
        Calculates the Pseudo Amino Acid Composition of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the Pseudo Amino Acid Composition for.

        lambda_ : int, default=10
            The lambda value to use for the calculation.

        weight : float, default=0.05
            The weight to use for the calculation.

        kwargs

        Returns
        -------
        list of float
            The Pseudo Amino Acid Composition of the sequence.
        """
        correlation_factors = [get_sequence_order_correlation_factor(sequence, i + 1) for i in range(lambda_)]
        correlation_factor = 1 + (weight * sum(correlation_factors))

        aac_counter = AminoAcidCompositionDescriptor()
        aac = aac_counter(sequence)
        paas = [round(aa / correlation_factor, 3) for aa in aac]

        lambdas = [round(weight * correlation_factors[i - 20] / correlation_factor * 100, 3)
                   for i in range(20, 20 + lambda_)]

        return paas + lambdas

    def get_features_out(self, lambda_: int = 10, **kwargs):
        paas = [f'PAAC_{aa}' for aa in AA_ALPHABET]
        lambdas = [f'PAAC_lambda_{i}' for i in range(1, lambda_ + 1)]
        return paas + lambdas


class AmphiphilicPseudoAminoAcidCompositionDescriptor(Descriptor):
    """
    A descriptor that returns the Amphiphilic Pseudo Amino Acid Composition of the sequence.
    """

    def __call__(self, sequence: str, lambda_: int = 10, weight: float = 0.05, **kwargs):
        """
        Calculates the Amphiphilic Pseudo Amino Acid Composition of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the Pseudo Amino Acid Composition for.

        lambda_ : int, default=10
            The lambda value to use for the calculation.

        weight : float, default=0.05
            The weight to use for the calculation.

        kwargs

        Returns
        -------
        list of float
            The Amphiphilic Pseudo Amino Acid Composition of the sequence.
        """
        correlation_factors = []

        for i in range(lambda_):
            correlation_factors.extend(get_sequence_order_correlation_factor_for_apaac(sequence, i + 1))

        correlation_factor = 1 + (weight * sum(correlation_factors))

        aac_counter = AminoAcidCompositionDescriptor()
        aac = aac_counter(sequence)
        paas = [round(aa / correlation_factor, 3) for aa in aac]

        lambdas = [round(weight * correlation_factors[i - 20] / correlation_factor * 100, 3)
                   for i in range(20, 20 + (2 * lambda_))]

        return paas + lambdas

    def get_features_out(self, lambda_: int = 10, **kwargs):
        paas = [f'APAAC_{aa}' for aa in AA_ALPHABET]
        lambdas = [f'APAAC_lambda_{i}' for i in range(1, (2 * lambda_) + 1)]
        return paas + lambdas
