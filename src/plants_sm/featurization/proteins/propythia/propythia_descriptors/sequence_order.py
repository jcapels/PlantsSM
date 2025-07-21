from typing import List

from ._constants import DISTANCE_1_POW_2, DISTANCE_2_POW_2, AA_ALPHABET
from ._utils import Descriptor, pairwise
from .amino_acid_composition import AminoAcidCompositionDescriptor


def sequence_order_coupling_number(protein_sequence: str,
                                   max_lag: int = 45,
                                   distance_matrix: dict = DISTANCE_1_POW_2) -> List[float]:
    """
    It computes the sequence order coupling numbers from 1 to max lag
    for a given protein sequence based on the distance matrix

    Parameters
    ----------
    protein_sequence : str
        A protein sequence

    max_lag : int
        Maximum lag to compute the sequence order coupling numbers

    distance_matrix : dict
        A distance matrix

    Returns
    -------
    sequence_order_coupling_number : List[float]
        List of sequence order coupling numbers
    """
    tau = []
    for i in range(max_lag):
        pairs = pairwise(protein_sequence, i + 1)
        tau.append(sum(distance_matrix[(a, b)] for a, b in pairs))
    return tau


class SequenceOrderCouplingNumbersDescriptor(Descriptor):
    """
    It calculates the Sequence Order Coupling Numbers of a sequence.
    """

    def __call__(self, sequence: str, max_lag: int = 45, **kwargs):
        """
        It calculates the Sequence Order Coupling Numbers of a sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the Sequence Order Coupling Numbers for.

        max_lag : int
            The maximum lag to consider. It should be inferior to the length of the sequence.

        kwargs

        Returns
        -------
        list of float
            the Sequence Order Coupling Numbers of a sequence.
        """
        if len(sequence) < max_lag:
            raise ValueError("The sequence length should be superior to the max_lag.")

        tau_sw = sequence_order_coupling_number(sequence, max_lag, distance_matrix=DISTANCE_1_POW_2)
        tau_grant = sequence_order_coupling_number(sequence, max_lag, distance_matrix=DISTANCE_2_POW_2)
        return tau_sw + tau_grant

    def get_features_out(self, max_lag: int = 45, **kwargs):
        return [f'tau_sw_{i}' for i in range(1, max_lag + 1)] + [f'tau_grant_{i}' for i in range(1, max_lag + 1)]


def quasi_sequence_order_1(protein_sequence: str,
                           max_lag: int = 30,
                           weight: float = 0.1,
                           distance_matrix: dict = DISTANCE_1_POW_2) -> List[float]:
    """
    It computes the quasi sequence order coupling numbers from 1 to max lag
    for a given protein sequence based on the distance matrix

    Parameters
    ----------
    protein_sequence : str
        A protein sequence

    max_lag : int
        Maximum lag to compute the quasi sequence order coupling numbers

    weight : float
        Weight of the amino acid composition descriptor

    distance_matrix : dict
        A distance matrix

    Returns
    -------
    quasi_sequence_order : List[float]
        List of quasi sequence order coupling numbers
    """
    sequence_order_factor = sum(sequence_order_coupling_number(protein_sequence, max_lag, distance_matrix))
    sequence_order_factor = 1 + (weight * sequence_order_factor)

    aac_descriptor = AminoAcidCompositionDescriptor()
    aac_cnt = aac_descriptor(protein_sequence)
    aac_keys = aac_descriptor.get_features_out()
    aac = dict(zip(aac_keys, aac_cnt))

    tau = []
    for i, aa in enumerate(AA_ALPHABET):
        tau.append(round(aac[aa] / sequence_order_factor, 6))
    return tau


def quasi_sequence_order_2(protein_sequence: str,
                           max_lag: int = 30,
                           weight: float = 0.1,
                           distance_matrix: dict = DISTANCE_1_POW_2) -> List[float]:
    """
    It computes the quasi sequence order coupling numbers from 1 to max lag
    for a given protein sequence based on the distance matrix

    Parameters
    ----------
    protein_sequence : str
        A protein sequence

    max_lag : int
        Maximum lag to compute the quasi sequence order coupling numbers

    weight : float
        Weight of the amino acid composition descriptor

    distance_matrix : dict
        A distance matrix

    Returns
    -------
    quasi_sequence_order : List[float]
        List of quasi sequence order coupling numbers
    """
    sequence_order_factors = sequence_order_coupling_number(protein_sequence, max_lag, distance_matrix)
    factor = 1 + (weight * sum(sequence_order_factors))

    tau = []
    for i in range(max_lag):
        tau.append(round(weight * sequence_order_factors[1] / factor, 6))
    return tau


class QuasiSequenceOrderDescriptor(Descriptor):
    """
    It calculates the Quasi-Sequence Order of a sequence.
    """

    def __call__(self, sequence: str, max_lag: int = 30, weight: float = 0.1, **kwargs):
        """
        It calculates the Quasi-Sequence Order of a sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the Quasi-Sequence Order for.

        max_lag : int
            The maximum lag to consider. It should be inferior to the length of the sequence.

        weight : float
            The weight to apply to the quasi-sequence order.

        kwargs

        Returns
        -------
        list of float
            the Quasi-Sequence Order of a sequence.
        """
        if len(sequence) < max_lag:
            raise ValueError("The sequence length should be superior to the max_lag.")

        tau_qso_1_sw = quasi_sequence_order_1(sequence, max_lag, distance_matrix=DISTANCE_1_POW_2)
        tau_qso_1_grant = quasi_sequence_order_1(sequence, max_lag, distance_matrix=DISTANCE_2_POW_2)
        tau_qso_2_sw = quasi_sequence_order_2(sequence, max_lag, distance_matrix=DISTANCE_1_POW_2)
        tau_qso_2_grant = quasi_sequence_order_2(sequence, max_lag, distance_matrix=DISTANCE_2_POW_2)
        return tau_qso_1_sw + tau_qso_2_sw + tau_qso_1_grant + tau_qso_2_grant

    def get_features_out(self, max_lag: int = 30, **kwargs):
        tau_qso_1_sw = [f"tau_qso_1_sw_{aa}" for aa in range(len(AA_ALPHABET))]
        tau_qso_1_grant = [f"tau_qso_1_grant_{aa}" for aa in range(len(AA_ALPHABET))]
        tau_qso_2_sw = [f"tau_qso_2_sw_{i}" for i in range(max_lag)]
        tau_qso_2_grant = [f"tau_qso_2_grant_{i}" for i in range(max_lag)]
        return tau_qso_1_sw + tau_qso_2_sw + tau_qso_1_grant + tau_qso_2_grant
