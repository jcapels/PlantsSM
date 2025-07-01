from collections import defaultdict, Counter
from statistics import mean

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

from ._constants import BONDS
from ._utils import Descriptor


class LengthDescriptor(Descriptor):
    """
    A descriptor that returns the length of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        return [float(len(sequence.strip()))]

    def get_features_out(self, **kwargs):
        return ["length"]


class ChargeDescriptor(Descriptor):
    """
    A descriptor that returns the charge of the sequence.
    """

    def __call__(self, sequence: str, ph: float = 7.4, amide: bool = False, **kwargs):
        """
        Calculates the charge of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the charge for.

        ph: float, default 7.4
            ph considered for the calculation.

        amide: bool, default False
            Considering as an amide protein sequence.
        kwargs

        Returns
        -------
        list of float
            The charge of the sequence.
        """
        desc = GlobalDescriptor(sequence)
        desc.calculate_charge(ph=ph, amide=amide)
        return [desc.descriptor[0][0]]

    def get_features_out(self, **kwargs):
        return ["charge"]


class ChargeDensityDescriptor(Descriptor):
    """
    A descriptor that returns the charge density of the sequence.
    """

    def __call__(self, sequence: str, ph: float = 7.4, amide: bool = False, **kwargs):
        """
        Calculates the charge density of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the charge density for.

        ph: float, default 7.4
            ph considered for the calculation.

        amide: bool, default False
            Considering as an amide protein sequence.
        kwargs

        Returns
        -------
        list of float
            The charge density of the sequence.
        """
        desc = GlobalDescriptor(sequence)
        desc.charge_density(ph=ph, amide=amide)
        return [desc.descriptor[0][0]]

    def get_features_out(self, **kwargs):
        return ["charge_density"]


class FormulaDescriptor(Descriptor):
    """
    A descriptor that returns the formula of the sequence.
    """

    def __call__(self, sequence: str, amide: bool = False, **kwargs):
        """
        Calculates the formula of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the formula for.

        amide: bool, default False
            Considering as an amide protein sequence.

        kwargs

        Returns
        -------
        list of float
            The formula of the sequence.
        """
        desc = GlobalDescriptor(sequence)
        desc.formula(amide)
        counts = desc.descriptor[0][0].split()
        counts = {atom[0]: int(atom[1:]) for atom in counts}

        formula = {'C': 0, 'H': 0, 'N': 0, 'O': 0, 'S': 0}
        formula.update(counts)
        return list(formula.values())

    def get_features_out(self, **kwargs):
        return ['Carbon', 'Hydrogen', 'Nitrogen', 'Oxygen', 'Sulfur']


class BondDescriptor(Descriptor):
    """
    A descriptor that returns the bond order of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the bond order of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the bond order for.

        kwargs

        Returns
        -------
        list of float
            The bond order of the sequence.
        """
        aa_counts = Counter(sequence.upper())

        bonds = defaultdict(int)
        for aa, cnt in aa_counts.items():
            if aa in BONDS:
                bonds['total'] += BONDS[aa]['total_bounds'] * cnt
                bonds['hydrogen'] += BONDS[aa]['hydrogen_bonds'] * cnt
                bonds['single'] += BONDS[aa]['single_bounds'] * cnt
                bonds['double'] += BONDS[aa]['double_bounds'] * cnt
        return list(bonds.values())

    def get_features_out(self, **kwargs):
        return ['total', 'hydrogen', 'single', 'double']


class MolecularWeightDescriptor(Descriptor):
    """
    A descriptor that returns the molecular weight of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the molecular weight of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the molecular weight for.

        kwargs

        Returns
        -------
        list of float
            The molecular weight of the sequence.
        """
        desc = GlobalDescriptor(sequence)
        desc.calculate_MW(amide=True)
        return [desc.descriptor[0][0]]

    def get_features_out(self, **kwargs):
        return ["molecular_weight"]


class GravyDescriptor(Descriptor):
    """
    A descriptor that returns the gravy of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the gravy of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the gravy for.

        kwargs

        Returns
        -------
        list of float
            The gravy of the sequence.
        """
        analysis = ProteinAnalysis(sequence)
        return [analysis.gravy()]

    def get_features_out(self, **kwargs):
        return ["gravy"]


class AromacityDescriptor(Descriptor):
    """
    A descriptor that returns the aromacity of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the aromacity of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the aromacity for.

        kwargs

        Returns
        -------
        list of float
            The aromacity of the sequence.
        """
        analysis = ProteinAnalysis(sequence)
        return [analysis.aromaticity()]

    def get_features_out(self, **kwargs):
        return ["aromaticity"]


class IsoelectricPointDescriptor(Descriptor):
    """
    A descriptor that returns the isoelectric point of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the isoelectric point of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the isoelectric point for.

        kwargs

        Returns
        -------
        list of float
            The isoelectric point of the sequence.
        """
        analysis = ProteinAnalysis(sequence)
        return [analysis.isoelectric_point()]

    def get_features_out(self, **kwargs):
        return ["isoelectric_point"]


class InstabilityIndexDescriptor(Descriptor):
    """
    A descriptor that returns the instability index of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the instability index of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the instability index for.

        kwargs

        Returns
        -------
        list of float
            The instability index of the sequence.
        """
        analysis = ProteinAnalysis(sequence)
        return [analysis.instability_index()]

    def get_features_out(self, **kwargs):
        return ["instability_index"]


class SecondaryStructureDescriptor(Descriptor):
    """
    A descriptor that returns the secondary structure of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the secondary structure of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the secondary structure for.

        kwargs

        Returns
        -------
        list of float
            The secondary structure of the sequence.
        """
        analysis = ProteinAnalysis(sequence)
        helix, turn, sheet = analysis.secondary_structure_fraction()
        return [helix, turn, sheet]

    def get_features_out(self, **kwargs):
        return ["secondary_structure_helix", "secondary_structure_turn", "secondary_structure_sheet"]


class MolarExtinctionCoefficientDescriptor(Descriptor):
    """
    A descriptor that returns the molar extinction coefficient of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the molar extinction coefficient of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the molar extinction coefficient for.

        kwargs

        Returns
        -------
        list of float
            The molar extinction coefficient of the sequence.
        """
        analysis = ProteinAnalysis(sequence)
        reduced, oxidized = analysis.molar_extinction_coefficient()
        return [reduced, oxidized]

    def get_features_out(self, **kwargs):
        return ["molar_extinction_coefficient_reduced", "molar_extinction_coefficient_oxidized"]


class FlexibilityDescriptor(Descriptor):
    """
    A descriptor that returns the flexibility of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the flexibility of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the flexibility for.

        kwargs

        Returns
        -------
        list of float
            The flexibility of the sequence.
        """
        # TODO: it should be confirmed if this is the correct way to calculate the flexibility.
        #  The original code was returning the list of all the flexibility values.
        #  However, the name of the feature cannot vary according to the sequence length.
        analysis = ProteinAnalysis(sequence)
        flexibility = analysis.flexibility()
        return [mean(flexibility)]

    def get_features_out(self, **kwargs):
        return ["flexibility"]


class AliphaticIndexDescriptor(Descriptor):
    """
    A descriptor that returns the aliphatic index of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the aliphatic index of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the aliphatic index for.

        kwargs

        Returns
        -------
        list of float
            The aliphatic index of the sequence.
        """
        desc = GlobalDescriptor(sequence)
        desc.aliphatic_index()
        return [desc.descriptor[0][0]]

    def get_features_out(self, **kwargs):
        return ["aliphatic_index"]


class BomanIndexDescriptor(Descriptor):
    """
    A descriptor that returns the Boman index of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the Boman index of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the Boman index for.

        kwargs

        Returns
        -------
        list of float
            The Boman index of the sequence.
        """
        desc = GlobalDescriptor(sequence)
        desc.boman_index()
        return [desc.descriptor[0][0]]

    def get_features_out(self, **kwargs):
        return ["boman_index"]


class HydrophobicRatioDescriptor(Descriptor):
    """
    A descriptor that returns the hydrophobic ratio of the sequence.
    """

    def __call__(self, sequence: str, **kwargs):
        """
        Calculates the hydrophobic ratio of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the hydrophobic ratio for.

        kwargs

        Returns
        -------
        list of float
            The hydrophobic ratio of the sequence.
        """
        desc = GlobalDescriptor(sequence)
        desc.hydrophobic_ratio()
        return [desc.descriptor[0][0]]

    def get_features_out(self, **kwargs):
        return ["hydrophobic_ratio"]


class ModlAMPMomentDescriptor(Descriptor):
    """
    A descriptor that returns the moment of the sequence.
    """

    def __call__(self, sequence: str,
                 window: int = 7,
                 scalename: str = 'Eisenberg',
                 angle: int = 100,
                 modality: str = 'max',
                 **kwargs):
        """
        Calculates the moment of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the moment for.

        scalename : str
            The name of the scale to use.

        angle : int
            The angle to use for the moment calculation.

        modality : str
            The modality to use for the moment calculation.

        kwargs

        Returns
        -------
        list of float
            The moment of the sequence.
        """
        desc = PeptideDescriptor(sequence, scalename)
        desc.calculate_moment(window, angle, modality)
        return [desc.descriptor[0][0]]

    def get_features_out(self, **kwargs):
        return ["moment"]


class ModlAMPGlobalDescriptor(Descriptor):
    """
    A descriptor that returns the global of the sequence.
    """

    def __call__(self, sequence: str,
                 window: int = 7,
                 scalename: str = 'Eisenberg',
                 modality: str = 'max',
                 **kwargs):
        """
        Calculates the global of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the global for.

        scalename : str
            The name of the scale to use.

        modality : str
            The modality to use for the global calculation.

        kwargs

        Returns
        -------
        list of float
            The global of the sequence.
        """
        desc = PeptideDescriptor(sequence, scalename)
        desc.calculate_global(window, modality)
        return [desc.descriptor[0][0]]

    def get_features_out(self, **kwargs):
        return ["global"]


class ModlAMPProfileDescriptor(Descriptor):
    """
    A descriptor that returns the profile of the sequence.
    """
    def __call__(self, sequence: str,
                 window: int = 7,
                 scalename: str = 'Eisenberg',
                 prof_type: str = 'uH',
                 **kwargs):
        """
        Calculates the profile of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the profile for.

        scalename : str
            The name of the scale to use.

        prof_type : str
            The type of profile to calculate.

        kwargs

        Returns
        -------
        list of float
            The profile of the sequence.
        """
        desc = PeptideDescriptor(sequence, scalename)
        desc.calculate_profile(prof_type, window)
        return list(desc.descriptor[0])

    def get_features_out(self, **kwargs):
        # profile descriptor returns only two values
        return ['profile_1', 'profile_2']


class ModlAMPArcDescriptor(Descriptor):
    """
    A descriptor that returns the arcs of the sequence.
    """
    def __call__(self, sequence: str,
                 window: int = 7,
                 scalename: str = 'peparc',
                 modality: str = 'max',
                 **kwargs):
        """
        Calculates the arcs of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the arcs for.

        scalename : str
            The name of the scale to use.

        modality : str
            The modality to use for the global calculation.

        kwargs

        Returns
        -------
        list of float
            The arcs of the sequence.
        """
        desc = PeptideDescriptor(sequence, scalename)
        desc.calculate_arc(modality)
        return list(desc.descriptor[0])

    def get_features_out(self, **kwargs):
        # arc descriptor returns only five values
        return [f'arc_{i}' for i in range(1, 5 + 1)]
