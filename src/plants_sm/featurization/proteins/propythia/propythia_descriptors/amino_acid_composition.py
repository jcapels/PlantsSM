from collections import Counter
from itertools import product

from ._constants import AA_ALPHABET
from ._utils import Descriptor


class AminoAcidCompositionDescriptor(Descriptor):
    """
    A descriptor that returns the Amino Acid Composition of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the Amino Acid Composition of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the isoelectric point for.

        kwargs

        Returns
        -------
        list of float
            The Amino Acid Composition of the sequence.
        """
        counts = Counter(sequence)
        return [round(counts.get(aa, 0) / len(sequence) * 100, 3) for aa in AA_ALPHABET]

    def get_features_out(self, **kwargs):
        return list(AA_ALPHABET)


class DipeptideCompositionDescriptor(Descriptor):
    """
    A descriptor that returns the Dipeptide Composition of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the Dipeptide Composition of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the Dipeptide Composition for.

        kwargs

        Returns
        -------
        list of float
            The Dipeptide Composition of the sequence.
        """
        counts = {a + b: 0 for a, b in product(AA_ALPHABET, AA_ALPHABET)}

        for a, b in zip(sequence, sequence[1:]):
            counts[a + b] += 1

        return [round(cnt / (len(sequence) - 1) * 100, 3) for cnt in counts.values()]

    def get_features_out(self, **kwargs):
        return [f'{aa_1}{aa_2}' for aa_1, aa_2 in product(AA_ALPHABET, AA_ALPHABET)]


class TripeptideCompositionDescriptor(Descriptor):
    """
    A descriptor that returns the Tripeptide Composition of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the Tripeptide Composition of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the Tripeptide Composition for.

        kwargs

        Returns
        -------
        list of float
            The Tripeptide Composition of the sequence.
        """
        counts = {(a, b, c): 0 for a, b, c in product(AA_ALPHABET, AA_ALPHABET, AA_ALPHABET)}

        for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
            counts[(a, b, c)] += 1

        return [round(cnt / (len(sequence) - 2) * 100, 3) for cnt in counts.values()]

    def get_features_out(self, **kwargs):
        return [f'{aa_1}{aa_2}{aa_3}' for aa_1, aa_2, aa_3 in product(AA_ALPHABET, AA_ALPHABET, AA_ALPHABET)]
