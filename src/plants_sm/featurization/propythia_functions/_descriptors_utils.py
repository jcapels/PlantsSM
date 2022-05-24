from typing import Dict, Any, List

from .descriptors_modlamp import GlobalDescriptor, PeptideDescriptor
from .bondcomp import boc_wp
from .aa_composition import calculate_aa_composition, \
    calculate_dipeptide_composition, get_spectrum_dict
from .pseudo_aac import get_pseudo_aac, get_a_pseudo_aac
from .autocorrelation import \
    calculate_normalized_moreau_broto_auto_total, calculate_moran_auto_total, calculate_geary_auto_total
from .ctd import calculate_ctd
from .quasi_sequence_order import \
    get_sequence_order_coupling_number_total, get_quasi_sequence_order
from .quasi_sequence_order import \
    get_sequence_order_coupling_numberp, get_quasi_sequence_orderp
from .conjoint_triad import calculate_conjoint_triad
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def adjuv_charge(protein_sequence: str, ph: float = 7.4, amide: bool = False) -> Dict[str, Any]:
    """
    Calculates charge of sequence (1 value) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence
    ph: float (default 7.4)
        ph to calculate charge
    amide: bool (default True)
        by default is not considered an amide protein sequence
    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    desc = GlobalDescriptor(protein_sequence)
    desc.calculate_charge(ph=ph, amide=amide)
    res['charge'] = desc.descriptor[0][0]
    return res


def adjuv_bond(protein_sequence: str) -> Dict[str, Any]:
    """
    This function gives the sum of the bond composition for each type of bond
    For bond composition four types of bonds are considered
    total number of bonds (including aromatic), hydrogen bond, single bond and double bond.

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    res.update(boc_wp(protein_sequence))
    return res


def adjuv_length(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates lenght of sequence (number of aa)

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {'length': float(len(protein_sequence.strip()))}
    return res


def adjuv_charge_density(protein_sequence: str, ph: float = 7.0, amide: bool = False) -> Dict[str, Any]:
    """
    Calculates charge density of sequence (1 value) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence
    ph: float (default 7.0)
        ph to calculate charge
    amide: bool (default True)
        by default is not considered an amide protein sequence
    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """

    res = {}
    desc = GlobalDescriptor(protein_sequence)
    desc.charge_density(ph, amide)
    res['charge_density'] = desc.descriptor[0][0]
    return res


def adjuv_formula(protein_sequence: str, amide: bool = False) -> Dict[str, Any]:
    """
    Calculates number of C,H,N,O and S of the aa of sequence (5 values) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence
    amide: bool (default True)
        by default is not considered an amide protein sequence
    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the 5 values of C,H,N,O and S
    """
    res = {}
    desc = GlobalDescriptor(protein_sequence)
    desc.formula(amide)
    formula = desc.descriptor[0][0].split()
    for atom in formula:
        if atom[0] == 'C':
            res['formulaC'] = int(atom[1:])
        if atom[0] == 'H':
            res['formulaH'] = int(atom[1:])
        if atom[0] == 'N':
            res['formulaN'] = int(atom[1:])
        if atom[0] == 'O':
            res['formulaO'] = int(atom[1:])
        if atom[0] == 'S':
            res['formulaS'] = int(atom[1:])
    # some formulas, specially S sometimes culd be a zero, to not transform iinto a nan in dataset
    if not res.get('formulaC'):
        res['formulaC'] = 0
    if not res.get('formulaH'):
        res['formulaH'] = 0
    if not res.get('formulaN'):
        res['formulaN'] = 0
    if not res.get('formulaO'):
        res['formulaO'] = 0
    if not res.get('formulaS'):
        res['formulaS'] = 0
    return res


def adjuv_mw(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates molecular weight of sequence (1 value) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """

    res = {}
    desc = GlobalDescriptor(protein_sequence)
    desc.calculate_MW(amide=True)
    res['MW_modlamp'] = desc.descriptor[0][0]
    return res


def adjuv_gravy(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates Gravy from sequence (1 value) from biopython

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """

    res = {}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Gravy'] = analysed_seq.gravy()
    return res


def adjuv_aromacity(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates Aromacity from sequence (1 value) from biopython

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """

    res = {}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Aromacity'] = analysed_seq.aromaticity()
    return res


def adjuv_isoelectric_point(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates Isolectric Point from sequence (1 value) from biopython

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """

    res = {}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['IsoelectricPoint'] = analysed_seq.isoelectric_point()
    return res


def adjuv_instability_index(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates Instability index from sequence (1 value) from biopython

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Instability_index'] = analysed_seq.instability_index()
    return res


def adjuv_sec_struct(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates the fraction of amino acids which tend to be in helix, turn or sheet (3 value) from biopython

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['SecStruct_helix'] = analysed_seq.secondary_structure_fraction()[0]  # helix
    res['SecStruct_turn'] = analysed_seq.secondary_structure_fraction()[1]  # turn
    res['SecStruct_sheet'] = analysed_seq.secondary_structure_fraction()[2]  # sheet
    return res


def adjuv_molar_extinction_coefficient(protein_sequence: str) -> Dict[str, Any]:  # [reduced, oxidized] # with
    # reduced cysteines / #
    # with disulfid bridges
    """
    Calculates the molar extinction coefficient (2 values) from biopython

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Molar_extinction_coefficient_reduced'] = analysed_seq.molar_extinction_coefficient()[0]  # reduced
    res['Molar_extinction_coefficient_oxidized'] = analysed_seq.molar_extinction_coefficient()[1]  # cys cys bounds
    return res


def adjuv_flexibility(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates the flexibility according to Vihinen, 1994 (return proteinsequencelenght-9 values ) from biopython

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """

    res = {}
    analysed_seq = ProteinAnalysis(protein_sequence)
    flexibility = analysed_seq.flexibility()
    for i in range(len(flexibility)):
        res['flexibility_' + str(i)] = flexibility[i]
    return res


def adjuv_aliphatic_index(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates aliphatic index of sequence (1 value) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    desc = GlobalDescriptor(protein_sequence)
    desc.aliphatic_index()
    res['aliphatic_index'] = desc.descriptor[0][0]
    return res


def adjuv_boman_index(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates boman index of sequence (1 value) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    desc = GlobalDescriptor(protein_sequence)
    desc.boman_index()
    res['bomanindex'] = desc.descriptor[0][0]
    return res


def adjuv_hydrophobic_ratio(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates hydrophobic ratio of sequence (1 value) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    desc = GlobalDescriptor(protein_sequence)
    desc.hydrophobic_ratio()
    res['hydrophobic_ratio'] = desc.descriptor[0][0]
    return res


################## AMINO ACID COMPOSITION ##################

def adjuv_aa_comp(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates amino acid compositon (20 values)  from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    res.update(calculate_aa_composition(protein_sequence))
    return res


def adjuv_dp_comp(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates dipeptide composition (400 values) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    res.update(calculate_dipeptide_composition(protein_sequence))
    return res


def adjuv_tp_comp(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates tripeptide composition (8000 values) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    res.update(get_spectrum_dict(protein_sequence))
    return res


################## PSEUDO AMINO ACID COMPOSITION ##################

def adjuv_paac(protein_sequence: str, lamda: int = 10, weight: float = 0.05) -> Dict[str, Any]:
    """
    Calculates Type I Pseudo amino acid composition (default is 30, depends on lamda) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    lamda: int
        reflects the rank of correlation and is a non-Negative integer, such as 10.
        should NOT be larger than the length of input protein sequence
        when lamda =0, the output of PseAA server is the 20-D amino acid composition

    weight: float
        weight on the additional PseAA components. with respect to the conventional AA components.
        The user can select any value within the region from 0.05 to 0.7 for the weight factor.

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the fractions of all PAAC (keys are the PAAC). Number of keys depends on lamda
    """
    res = {}
    res.update(get_pseudo_aac(protein_sequence, lamda=lamda, weight=weight))
    return res


def adjuv_paac_p(protein_sequence: str, lamda: int = 10, weight: float = 0.05, AAP: List[Dict[Any, Any]] = None) -> \
        Dict[str, Any]:
    """
    Calculates Type I Pseudo amino acid composition for a given property (default is 30, depends on lamda) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    lamda: int
        reflects the rank of correlation and is a non-Negative integer, such as 10.
        should NOT be larger than the length of input protein sequence
        when lamda =0, the output of PseAA server is the 20-D amino acid composition

    weight: float
        weight on the additional PseAA components. with respect to the conventional AA components.
        The user can select any value within the region from 0.05 to 0.7 for the weight factor.

    AAP: List[Dict[Any, Any]]
        list of properties. each of which is a dict form.
        PseudoAAC._Hydrophobicity,PseudoAAC._hydrophilicity, PseudoAAC._residuemass,PseudoAAC._pK1,PseudoAAC._pK2,
        PseudoAAC._pI

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the fractions of all PAAC (keys are the PAAC). Number of keys depends on lamda
    """
    res = {}
    res.update(get_pseudo_aac(protein_sequence, lamda=lamda, weight=weight, AAP=AAP))
    return res


def adjuv_apaac(protein_sequence: str, lamda: int = 10, weight: float = 0.5) -> Dict[str, Any]:
    """
    Calculates Type II Pseudo amino acid composition - Amphiphilic (default is 30, depends on lamda) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    lamda: int
        reflects the rank of correlation and is a non-Negative integer, such as 10.
        should NOT be larger than the length of input protein sequence
        when lamda =0, the output of PseAA server is the 20-D amino acid composition

    weight: float
        weight on the additional PseAA components. with respect to the conventional AA components.
        The user can select any value within the region from 0.05 to 0.7 for the weight factor.

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the fractions of all PAAC (keys are the PAAC). Number of keys depends on lamda
    """
    res = {}
    res.update(get_a_pseudo_aac(protein_sequence, lamda=lamda, weight=weight))
    return res


# ################# AUTOCORRELATION DESCRIPTORS ##################

def adjuv_moreau_broto_auto(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates Normalized Moreau-Broto autocorrelation (240 values) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    res.update(calculate_normalized_moreau_broto_auto_total(protein_sequence))
    return res


def adjuv_moran_auto(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates  Moran autocorrelation (240 values) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    res.update(calculate_moran_auto_total(protein_sequence))
    return res


def adjuv_geary_auto(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates  Geary autocorrelation (240 values) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    res.update(calculate_geary_auto_total(protein_sequence))
    return res


# ################# COMPOSITION, TRANSITION, DISTRIBUTION ##################

def adjuv_ctd(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates the Composition Transition Distribution descriptors (147 values) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    res.update(calculate_ctd(protein_sequence))
    return res


# ################# CONJOINT TRIAD ##################

def adjuv_conj_t(protein_sequence: str) -> Dict[str, Any]:
    """
    Calculates the Conjoint Triad descriptors (343 descriptors) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    res = {}
    res.update(calculate_conjoint_triad(protein_sequence))
    return res


# #################  SEQUENCE ORDER  ##################

def adjuv_socn(protein_sequence: str, maxlag: int = 45) -> Dict[str, Any]:
    """
    Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    maxlag: int
        maximum lag. Smaller than length of the protein

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence
    """
    if maxlag > len(protein_sequence):
        raise Exception(
            "Parameter max-lag must be smaller than length of the protein")
    res = {}
    res.update(get_sequence_order_coupling_number_total(protein_sequence, maxlag=maxlag))
    return res


def adjuv_socn_p(protein_sequence: str, maxlag: int = 45, distance_matrix: Dict = None) -> Dict[str, Any]:
    """
    Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    maxlag: int
        maximum lag. Smaller than length of the protein

    distance_matrix: Dict
        dict form containing 400 distance values

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the features names as keys and the values for the given protein sequence (90 descriptors)
    """
    if maxlag > len(protein_sequence):
        raise Exception("Parameter maxlag must be smaller than length of the protein")
    res = {}
    res.update(get_sequence_order_coupling_numberp(protein_sequence, maxlag=maxlag, distancematrix=distance_matrix))
    return res


def adjuv_qso(protein_sequence: str, maxlag: int = 30, weight: float = 0.1) -> Dict[str, Any]:
    """
    Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    maxlag: int
        maximum lag. Smaller than length of the protein

    weight: float
        weight on the additional PseAA components. with respect to the conventional AA components.
        The user can select any value within the region from 0.05 to 0.7 for the weight factor.

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the descriptors (100 descriptors)
    """
    if maxlag > len(protein_sequence):
        raise Exception("Parameter maxlag must be smaller than length of the protein")
    res = {}
    res.update(get_quasi_sequence_order(protein_sequence, maxlag=maxlag, weight=weight))
    return res


def adjuv_qso_p(protein_sequence: str, maxlag: int = 30, weight: float = 0.1, distance_matrix: Dict = None) -> \
        Dict[str, Any]:
    """
    Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    maxlag: int
        maximum lag. Smaller than length of the protein

    weight: float
        weight on the additional PseAA components. with respect to the conventional AA components.
        The user can select any value within the region from 0.05 to 0.7 for the weight factor.

    distance_matrix: Dict
        dict form containing 400 distance values

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the descriptors (100 descriptors)
    """
    if maxlag > len(protein_sequence):
        raise Exception("Parameter maxlag must be smaller than length of the protein")
    res = {}
    res.update(get_quasi_sequence_orderp(protein_sequence, maxlag=maxlag, weight=weight,
                                         distancematrix=distance_matrix))
    return res


# ################# BASE CLASS PEPTIDE DESCRIPTOR ##################

def adjuv_calculate_moment(protein_sequence: str, window: int = 1000, angle: int = 100, modality: str = 'max',
                           scale_name: str = 'Eisenberg') -> Dict[str, Any]:
    """
    Calculates moment of sequence (1 value) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    window: int
        amino acid window in which to calculate the moment. If the sequence is shorter than the window,
        the length of the sequence is taken

    angle: float
        angle in which to calculate the moment. 100 for alpha helices, 180 for beta sheets

    modality: str
        maximum (max), mean (mean) or both (all) hydrophobic moment

    scale_name: str
        name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values.
        By default Eisenberg.

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the descriptors (1 value)
    """
    res = {}
    AMP = PeptideDescriptor(protein_sequence, scale_name)
    AMP.calculate_moment(window, angle, modality)
    res['moment'] = AMP.descriptor[0][0]
    return res


def adjuv_calculate_global(protein_sequence: str, window: int = 1000, modality: str = 'max',
                           scale_name: str = 'Eisenberg') -> Dict[str, Any]:
    """
    Calculates a global / window averaging descriptor value of a given AA scale of sequence (1 value) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    window: int
        amino acid window in which to calculate the moment. If the sequence is shorter than the window,
        the length of the sequence is taken

    modality: str
        maximum (max), mean (mean) or both (all) hydrophobic moment

    scale_name: str
        name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values.
        By default Eisenberg.

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the descriptors (1 value)
    """
    res = {}
    AMP = PeptideDescriptor(protein_sequence, scale_name)
    AMP.calculate_global(window, modality)
    res['global'] = AMP.descriptor[0][0]
    return res


def adjuv_calculate_profile(protein_sequence: str, prof_type: str = 'uH', window: int = 7,
                            scale_name: str = 'Eisenberg') -> Dict[str, Any]:
    """
    Calculates hydrophobicity or hydrophobic moment profiles for given sequences and fitting for slope and intercep
    (2 values) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    window: int
        amino acid window in which to calculate the moment. If the sequence is shorter than the window,
        the length of the sequence is taken

    prof_type: str
        prof_type of profile, ‘H’ for hydrophobicity or ‘uH’ for hydrophobic moment

    scale_name: str
        name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values.
        By default Eisenberg.

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the descriptors (2 value)
    """
    res = {}
    AMP = PeptideDescriptor(protein_sequence, scale_name)
    AMP.calculate_profile(prof_type, window)
    desc = AMP.descriptor[0]
    for i in range(len(desc)):
        res['profile_' + str(i)] = desc[i]
    return res


def adjuv_calculate_arc(protein_sequence: str, modality: str = "max", scale_name: str = 'peparc') -> Dict[str, Any]:
    """
    Calculates arcs as seen in the helical wheel plot. Use for binary amino acid scales only (5 values) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    modality: str
        maximum (max), mean (mean) or both (all) hydrophobic moment

    scale_name: str (optional)
        name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values.
        By default peparc.

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with the descriptors (5 value)
    """
    res = {}
    arc = PeptideDescriptor(protein_sequence, scale_name)
    arc.calculate_arc(modality)
    desc = arc.descriptor[0]
    for i in range(len(desc)):
        res['arc_' + str(i)] = desc[i]
    return res


def adjuv_calculate_autocorr(protein_sequence: str, window: int = 7, scale_name: str = 'Eisenberg') -> Dict[str, Any]:
    """
    Calculates autocorrelation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    window: int
        correlation window for descriptor calculation in a sliding window approach

    scale_name: str (optional)
        name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values.
        By default Eisenberg.

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with values of auto correlation
    """
    res = {}
    AMP = PeptideDescriptor(protein_sequence, scale_name)
    AMP.calculate_autocorr(window)
    desc = AMP.descriptor[0]
    for i in range(len(desc)):
        res['autocorr_' + str(i)] = desc[i]
    return res


def adjuv_calculate_crosscorr(protein_sequence: str, window: int = 7, scale_name: str = 'Eisenberg') -> Dict[str, Any]:
    """
    Calculates cross correlation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp

    Parameters
    ----------
    protein_sequence: str
        protein sequence

    window: int
        correlation window for descriptor calculation in a sliding window approach

    scale_name: str (optional)
        name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values.
        By default Eisenberg.

    Returns
    -------
    dictionary: Dict[str, Any]
        dictionary with values of cross correlation
    """
    res = {}
    AMP = PeptideDescriptor(protein_sequence, scale_name)
    AMP.calculate_crosscorr(window)
    desc = AMP.descriptor[0]
    for i in range(len(desc)):
        res['crosscorr_' + str(i)] = desc[i]
    return res
