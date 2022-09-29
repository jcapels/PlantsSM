import math

from ._utils import Descriptor
from ._constants import CTD_HYDROPHOBICITY_TABLE, CTD_NORMALIZED_VDWV_TABLE, CTD_POLARITY_TABLE, CTD_CHARGE_TABLE, \
    CTD_SECONDARY_STRUCTURE_TABLE, CTD_SOLVENT_ACCESSIBILITY_TABLE, POLARIZABILITY_TABLE


_CTD_TABLES = [CTD_HYDROPHOBICITY_TABLE, CTD_NORMALIZED_VDWV_TABLE, CTD_POLARITY_TABLE, CTD_CHARGE_TABLE,
               CTD_SECONDARY_STRUCTURE_TABLE, CTD_SOLVENT_ACCESSIBILITY_TABLE, POLARIZABILITY_TABLE]

_CTD_TABLE_NAMES = ['hydrophobicity', 'normalized_vdwv', 'polarity', 'charge', 'secondary_structure',
                    'solvent_accessibility', 'polarizability']


def calculate_composition(sequence, table):
    """
    It calculates the composition of a sequence.
    """
    n = len(sequence)
    translated_sequence = str.translate(sequence, table)
    c1 = round(float(translated_sequence.count('1')) / n, 3)
    c2 = round(float(translated_sequence.count('2')) / n, 3)
    c3 = round(float(translated_sequence.count('3')) / n, 3)
    return c1, c2, c3


def calculate_transition(sequence, table):
    """
    It calculates the transition of a sequence.
    """
    n = len(sequence)
    translated_sequence = str.translate(sequence, table)
    t12 = round(float(translated_sequence.count('12') + translated_sequence.count('21')) / (n - 1), 3)
    t13 = round(float(translated_sequence.count('13') + translated_sequence.count('31')) / (n - 1), 3)
    t23 = round(float(translated_sequence.count('23') + translated_sequence.count('32')) / (n - 1), 3)
    return t12, t13, t23


def calculate_distribution(sequence, table):
    """
    It calculates the distribution of a sequence.
    """
    n = len(sequence)
    translated_sequence = str.translate(sequence, table)

    result = []
    for i in ('1', '2', '3'):
        indexes = [j for j, char in enumerate(translated_sequence) if char == i]
        cnt = len(indexes)
        if cnt > 0:
            # it comprehends D_i_000, D_i_025, D_i_050, D_i_075, D_i_100
            to_extend = [
                round(float(indexes[0]) / n * 100, 3),
                round(float(indexes[int(math.floor(cnt * 0.25)) - 1]) / n * 100, 3),
                round(float(indexes[int(math.floor(cnt * 0.5)) - 1]) / n * 100, 3),
                round(float(indexes[int(math.floor(cnt * 0.75)) - 1]) / n * 100, 3),
                round(float(indexes[-1]) / n * 100, 3)
            ]
            result.extend(to_extend)

        else:
            # it comprehends D_i_000, D_i_025, D_i_050, D_i_075, D_i_100
            result.extend([0, 0, 0, 0, 0])

    return result


class CompositionTransitionDistributionDescriptor(Descriptor):
    """
    It calculates the composition, transition and distribution of a sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        It calculates the composition, transition and distribution of a sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the composition, transition and distribution for.

        kwargs

        Returns
        -------
        list of float
            the composition, transition and distribution of a sequence.
        """
        result = []
        for fn in (calculate_composition, calculate_transition, calculate_distribution):
            for table in _CTD_TABLES:
                result.extend(fn(sequence, table))
        return result

    def get_features_out(self, **kwargs):
        result = []
        for suffix in ('C', 'T', 'D'):

            if suffix == 'C':
                suffixes = [f'{suffix}_1', f'{suffix}_2', f'{suffix}_3']

            elif suffix == 'T':
                suffixes = [f'{suffix}_12', f'{suffix}_13', f'{suffix}_23']

            else:
                suffixes = [f'{suffix}_{i}_{j}' for i in ('1', '2', '3')
                            for j in ('000', '025', '050', '075', '100')]

            for name in _CTD_TABLE_NAMES:
                to_extend = [f'{name}_{suffix}' for suffix in suffixes]
                result.extend(to_extend)

        return result
