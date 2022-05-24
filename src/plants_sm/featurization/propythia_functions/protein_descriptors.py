"""
##############################################################################

A class  used for computing different types of protein descriptors parallelized.
It contains descriptors from packages pydpi, biopython, pfeature and modlamp.

Authors: Ana Marta Sequeira, Miguel Barros

Date: 05/2019 ALTERED 03/2022

Email:

##############################################################################
"""
import pandas as pd

from plants_sm.featurization.propythia_functions._descriptors_utils import adjuv_length, adjuv_charge, \
    adjuv_charge_density, adjuv_formula, adjuv_bond, adjuv_mw, adjuv_gravy, adjuv_aromacity, adjuv_isoelectric_point, \
    adjuv_instability_index, adjuv_sec_struct, adjuv_molar_extinction_coefficient, adjuv_flexibility, \
    adjuv_aliphatic_index, adjuv_boman_index, adjuv_hydrophobic_ratio, adjuv_aa_comp, adjuv_dp_comp, adjuv_tp_comp, \
    adjuv_paac, adjuv_paac_p, adjuv_apaac, adjuv_moreau_broto_auto, adjuv_moran_auto, adjuv_geary_auto, adjuv_ctd, \
    adjuv_conj_t, adjuv_socn, adjuv_socn_p, adjuv_qso, adjuv_qso_p, adjuv_calculate_moment, adjuv_calculate_global, \
    adjuv_calculate_profile, adjuv_calculate_arc, adjuv_calculate_autocorr, adjuv_calculate_crosscorr


class PropythiaDescriptors:

    @staticmethod
    def get_length(protein_sequence: str) -> pd.DataFrame:
        """
        Calculates length of sequence (number of aa)

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the value of length for each sequence in the dataset
        """
        res = adjuv_length(protein_sequence=protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_charge(protein_sequence: str, ph: float = 7.4, amide: bool = False):
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
        dataframe: pd.DataFrame
            Dataframe with the value of charge for each sequence in the dataset
        """

        res = adjuv_charge(protein_sequence, ph, amide)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_charge_density(protein_sequence: str, ph: float = 7.0, amide: bool = False):
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
        dataframe: pd.DataFrame
            Dataframe with the value of charge density for each sequence in the dataset
        """

        res = adjuv_charge_density(protein_sequence, ph, amide)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_formula(protein_sequence: str, amide: bool = False):
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
        dataframe: pd.DataFrame
            Dataframe with the 5 values of C,H,N,O and S for each sequence in the dataset
        """

        res = adjuv_formula(protein_sequence, amide)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_bond(protein_sequence: str):
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
        dataframe: pd.DataFrame
            Dataframe with 4 values for each sequence in the dataset
        """

        res = adjuv_bond(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_molecular_weight(protein_sequence: str):
        """
        Calculates molecular weight of sequence (1 value) from modlamp

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the value of molecular weight for each sequence in the dataset
        """

        res = adjuv_mw(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_gravy(protein_sequence):
        """
        Calculates Gravy from sequence (1 value) from biopython

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the value of gravy for each sequence in the dataset

        """

        res = adjuv_gravy(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_aromacity(protein_sequence: str):
        """
        Calculates Aromacity from sequence (1 value) from biopython

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the value of aromacity for each sequence in the dataset
        """

        res = adjuv_aromacity(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_isoelectric_point(protein_sequence: str):
        """
        Calculates Isolectric Point from sequence (1 value) from biopython

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the value of Isolectric Point for each sequence in the dataset
        """

        res = adjuv_isoelectric_point(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_instability_index(protein_sequence: str):
        """
        Calculates Instability index from sequence (1 value) from biopython

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the value of Instability index  for each sequence in the dataset
        """

        res = adjuv_instability_index(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_secondary_structure(protein_sequence: str):
        """
        Calculates the fraction of amino acids which tend to be in helix, turn or sheet (3 value) from biopython

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the 3 value of helix, turn, sheet for each sequence in the dataset
        """

        res = adjuv_sec_struct(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_molar_extinction_coefficient(protein_sequence: str):
        # [reduced, oxidized] # with reduced cysteines / # with disulfid bridges
        """
        Calculates the molar extinction coefficient (2 values) from biopython

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the value of reduced cysteins and oxidized (with disulfid bridges) for each sequence in the
            dataset
        """

        res = adjuv_molar_extinction_coefficient(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_flexibility(protein_sequence: str):
        """
        Calculates the flexibility according to Vihinen, 1994 (return proteinsequencelenght-9 values ) from biopython

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with proteinsequencelenght-9 values of flexiblity for each sequence in the dataset
        """

        res = adjuv_flexibility(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_aliphatic_index(protein_sequence: str):
        """
        Calculates aliphatic index of sequence (1 value) from modlamp

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the value of aliphatic index for each sequence in the dataset
        """

        res = adjuv_aliphatic_index(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_boman_index(protein_sequence: str):
        """
        Calculates boman index of sequence (1 value) from modlamp

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the value of boman index for each sequence in the dataset
        """

        res = adjuv_boman_index(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_hydrophobic_ratio(protein_sequence: str):
        """
        Calculates hydrophobic ratio of sequence (1 value) from modlamp

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the value of hydrophobic ratio for each sequence in the dataset
        """

        res = adjuv_hydrophobic_ratio(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    ################## AMINO ACID COMPOSITION ##################

    @staticmethod
    def get_aa_comp(protein_sequence: str):
        """
        Calculates amino acid compositon (20 values)  from pydpi

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the fractions of all 20 aa(keys are the aa) for each sequence in the dataset
        """

        res = adjuv_aa_comp(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_dp_comp(protein_sequence: str):
        """
        Calculates dipeptide composition (400 values) from pydpi

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the fractions of all 400 possible combiinations of 2 aa for each sequence in the dataset
        """

        res = adjuv_dp_comp(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_tp_comp(protein_sequence: str):
        """
        Calculates tripeptide composition (8000 values) from pydpi

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the fractions of all 8000 possible combinations of 3 aa for each sequence in the dataset
        """

        res = adjuv_tp_comp(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    ################## PSEUDO AMINO ACID COMPOSITION ##################

    @staticmethod
    def get_paac(protein_sequence: str, lamda: int = 10, weight: float = 0.05):
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
        dataframe: pd.DataFrame
            Dataframe with the fractions of all PAAC (keys are the PAAC) for each sequence in the dataset. Number of
            keys depends on lamda
        """
        res = adjuv_paac(protein_sequence, lamda, weight)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_paac_p(protein_sequence: str, lamda: int = 10, weight: float = 0.05, AAP=None):
        """
        Calculates Type I Pseudo amino acid composition for a given property (default is 30, depends on lamda)
        from pydpi

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
        dataframe: pd.DataFrame
            Dataframe with the fractions of all PAAC(keys are the PAAC) for each sequence in the dataset.
            Number of keys depends on lamda
        """

        res = adjuv_paac_p(protein_sequence, lamda, weight, AAP)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_apaac(protein_sequence: str, lamda: int = 10, weight: float = 0.5):
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
        dataframe: pd.DataFrame
            Dataframe with the fractions of all PAAC(keys are the PAAC) for each sequence in the dataset.
            Number of keys depends on lamda
        """

        res = adjuv_apaac(protein_sequence, lamda, weight)
        res = pd.DataFrame(res, index=[0])
        return res

    # ################# AUTOCORRELATION DESCRIPTORS ##################

    @staticmethod
    def get_moreau_broto_auto(protein_sequence: str):
        """
        Calculates Normalized Moreau-Broto autocorrelation (240 values) from pydpi

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the 240 descriptors for each sequence in the dataset
        """

        res = adjuv_moreau_broto_auto(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_moran_auto(protein_sequence: str):
        """
        Calculates  Moran autocorrelation (240 values) from pydpi

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the 240 descriptors for each sequence in the dataset

        """

        res = adjuv_moran_auto(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_geary_auto(protein_sequence: str):
        """
        Calculates  Geary autocorrelation (240 values) from pydpi

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the 240 descriptors for each sequence in the dataset
        """

        res = adjuv_geary_auto(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    # ################# COMPOSITION, TRANSITION, DISTRIBUTION ##################

    @staticmethod
    def get_ctd(protein_sequence: str):
        """
        Calculates the Composition Transition Distribution descriptors (147 values) from pydpi

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the 147 descriptors for each sequence in the dataset
        """

        res = adjuv_ctd(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    # ################# CONJOINT TRIAD ##################

    @staticmethod
    def get_conjoint_triad(protein_sequence: str):
        """
        Calculates the Conjoint Triad descriptors (343 descriptors) from pydpi

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with the 343 descriptors for each sequence in the dataset
        """

        res = adjuv_conj_t(protein_sequence)
        res = pd.DataFrame(res, index=[0])
        return res

    # #################  SEQUENCE ORDER  ##################

    @staticmethod
    def get_socn(protein_sequence: str, maxlag: int = 45):
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
        dataframe: pd.DataFrame
            Dataframe with the descriptors (90 descriptors) for each sequence in the dataset
        """

        res = adjuv_socn(protein_sequence, maxlag=maxlag)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_socn_p(protein_sequence: str, maxlag: int = 45, distance_matrix=None):
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
        dataframe: pd.DataFrame
            Dataframe with the descriptors (90 descriptors) for each sequence in the dataset
        """

        res = adjuv_socn_p(protein_sequence, maxlag, distance_matrix)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_qso(protein_sequence: str, maxlag: int = 30, weight: float = 0.1):
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
        dataframe: pd.DataFrame
            Dataframe with the descriptors (100 descriptors) for each sequence in the dataset
        """

        res = adjuv_qso(protein_sequence, maxlag, weight)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def get_qso_p(protein_sequence: str, maxlag: int = 30, weight: float = 0.1, distance_matrix=None):
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
        dataframe: pd.DataFrame
            Dataframe with the descriptors (100 descriptors)
        """

        res = adjuv_qso_p(protein_sequence, maxlag, weight, distance_matrix)
        res = pd.DataFrame(res, index=[0])
        return res

    # ################# BASE CLASS PEPTIDE DESCRIPTOR ##################

    """amino acid descriptor scales available are the ones from modlamo. 
    For more information please check: https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor
    amino acid sclaes include AASI, argos, bulkiness, charge_phys, charge_acid, eisenberg and others."""

    @staticmethod
    def calculate_moment(protein_sequence: str, window: int = 1000, angle: int = 100, modality: str = 'max',
                         scale_name: str = 'Eisenberg'):
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
            https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor
            values.
            By default Eisenberg.

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with one value of moment for each sequence in the dataset
        """
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")
        if angle != 100 and angle != 180: raise Exception(
            "Parameter angle must be 100 (alpha helices) or 180 (beta sheets)")
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")

        res = adjuv_calculate_moment(protein_sequence, window, angle, modality, scale_name)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def calculate_global(protein_sequence: str, window: int = 1000, modality: str = 'max',
                         scale_name: str = 'Eisenberg'):
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
        dataframe: pd.DataFrame
            Dataframe with one value for each sequence in the dataset
        """
        if modality != 'max' and modality != 'mean' and modality != 'all':
            raise ValueError("Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")

        res = adjuv_calculate_global(protein_sequence, window, modality, scale_name)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def calculate_profile(protein_sequence: str, prof_type: str = 'uH', window: int = 7, scale_name: str = 'Eisenberg'):
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
        dataframe: pd.DataFrame
            Dataframe with two value for each sequence in the dataset
        """
        if prof_type != 'H' and prof_type != 'uH': raise Exception(
            "Parameter prof_type must be 'H' (hydrophobicity) or 'uH' (hydrophobic)")

        res = adjuv_calculate_profile(protein_sequence, prof_type, window, scale_name)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def calculate_arc(protein_sequence: str, modality: str = "max", scale_name: str = 'peparc'):
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
            https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor
            values.
            By default peparc.

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with 5 values for each sequence in the dataset
        """
        if modality != 'max' and modality != 'mean' and modality != 'all':
            raise ValueError("Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")

        res = adjuv_calculate_arc(protein_sequence, modality, scale_name)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def calculate_autocorrelation(protein_sequence: str, window: int = 7, scale_name: str = 'Eisenberg'):
        """
        Calculates autocorrelation of amino acid values for a given descriptor scale ( variable >>>>>>values)
        from modlamp

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        window: int
            correlation window for descriptor calculation in a sliding window approach

        scale_name: str (optional)
            name of the amino acid scale (one in
            https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the
            descriptor values.
            By default Eisenberg.

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with values of autocorrelation for each sequence in the dataset
        """

        res = adjuv_calculate_autocorr(protein_sequence, window, scale_name)
        res = pd.DataFrame(res, index=[0])
        return res

    @staticmethod
    def calculate_crosscorrelation(protein_sequence: str, window: int = 7, scale_name: str = 'Eisenberg'):
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
        dictionary: pd.DataFrame
            Dataframe with values of crosscorrelation for each sequence in the dataset
        """

        res = adjuv_calculate_crosscorr(protein_sequence, window, scale_name)
        res = pd.DataFrame(res, index=[0])
        return res

    # ################# GET ALL FUNCTIONS ##################

    def get_all_physicochemical_properties(self, protein_sequence: str, ph: float = 7, amide: bool = False):
        """
        Calculate all 15 geral descriptors functions derived from biopython and modlpam

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
        dataframe: pd.DataFrame
            Dataframe with variable number of descriptors for each sequence in the dataset
        """
        result = self.get_length(protein_sequence)
        result = pd.concat((result, self.get_charge(protein_sequence, ph, amide)), axis=1)
        result = pd.concat((result, self.get_charge_density(protein_sequence, ph, amide)), axis=1)
        result = pd.concat((result, self.get_formula(protein_sequence, amide)), axis=1)
        result = pd.concat((result, self.get_bond(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_molecular_weight(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_gravy(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_aromacity(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_isoelectric_point(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_instability_index(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_secondary_structure(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_molar_extinction_coefficient(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_aliphatic_index(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_boman_index(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_hydrophobic_ratio(protein_sequence)), axis=1)
        return result

    def get_all_aac(self, protein_sequence: str):
        """
        Calculate all descriptors from Amino Acid Composition

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with values from AAC, DPC and TP for each sequence in the dataset
        """
        result = self.get_aa_comp(protein_sequence)
        result = pd.concat((result, self.get_dp_comp(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_tp_comp(protein_sequence)), axis=1)
        return result

    def get_all_paac(self, protein_sequence: str, lamda_paac: int = 10, weight_paac: float = 0.05,
                     lamda_apaac: int = 10,
                     weight_apaac: float = 0.05):
        """
        Calculate all descriptors from Pseudo Amino Acid Composition

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        lamda_paac: int
            parameter for PAAC default 10

        weight_paac: float
            parameter for PAAC default 0.05

        lamda_apaac: int
            parameter for APAAC default 10

        weight_apaac: float
            parameter for APAAC default 0.05

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with values from PAAC and APAAC  for each sequence in the dataset
        """
        result = self.get_paac(protein_sequence, lamda_paac, weight_paac)
        result = pd.concat((result, self.get_apaac(protein_sequence, lamda_apaac, weight_apaac)), axis=1)
        return result

    def get_all_sequence_order(self, protein_sequence: str, maxlag_socn: int = 45, maxlag_qso: int = 30,
                               weight_qso: float = 0.1):
        """
        Calculate all values for sequence order descriptors

        :param maxlag_socn: parameter for SOCN default 45
        :param maxlag_qso: parameter for QSO default 30
        :param weight_qso: parameter for QSO default 0.1
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with values for quasi sequence order and sequence order couplig numbers for each sequence
            in the dataset

        Calculate all descriptors from Pseudo Amino Acid Composition

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        maxlag_socn: int
            parameter for SOCN default 45

        maxlag_qso: int
            parameter for QSO default 30

        weight_qso: float
            parameter for QSO default 0.1

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with values for quasi sequence order and sequence order couplig numbers for each sequence
            in the dataset
        """
        result = self.get_socn(protein_sequence, maxlag_socn)
        result = pd.concat((result, self.get_qso(protein_sequence, maxlag_qso, weight_qso)), axis=1)
        return result

    def get_all_correlation(self, protein_sequence: str):
        """
        Calculate all descriptors from Autocorrelation

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe containing values for the functions Moreau Broto, Moran and Geary autocorrelation for each
            sequence in the dataset
        """
        result = self.get_moreau_broto_auto(protein_sequence)
        result = pd.concat((result, self.get_moran_auto(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_geary_auto(protein_sequence)), axis=1)
        return result

    def get_all_base_class(self, protein_sequence: str, window: int = 7, scale_name: str = 'Eisenberg',
                           scale_name_arc: str = 'peparc',
                           angle: int = 100, modality: str = 'max',
                           prof_type: str = 'uH'):
        """
        Calculate all functions from Base class

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        window: int
            size of sliding window used amino acid window. If the sequence is shorter than the window,
            the length of the sequence is taken.

        scale_name: str
            name of the amino acid scale (one in
            https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the
            descriptor values. By default Eisenberg.

        scale_name_arc: str
            name of the amino acid scale (one in
            https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor
            values only binary scales. By default peparc.

        angle: int
            angle in which to calculate the moment. 100 for alpha helices, 180 for beta sheets

        modality: str
            maximum (max), mean (mean) or both (all) hydrophobic moment

        prof_type: str
            prof_type of profile, ‘H’ for hydrophobicity or ‘uH’ for hydrophobic moment

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with all 6 base class peptide descriptors (the value is variable) for each sequence in the dataset
        """
        if prof_type != 'H' and prof_type != 'uH': raise Exception(
            "Parameter prof_type must be 'H' (hydrophobicity) or 'uH' (hydrophobic)")
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")
        if angle != 100 and angle != 180: raise Exception(
            "Parameter angle must be 100 (alpha helices) or 180 (beta sheets)")

        result = self.calculate_autocorrelation(protein_sequence, window, scale_name)
        result = pd.concat((result, self.calculate_crosscorrelation(protein_sequence, window, scale_name)),
                           axis=1)
        result = pd.concat((result, self.calculate_moment(protein_sequence, window, angle, modality, scale_name)),
                           axis=1)
        result = pd.concat((result, self.calculate_global(protein_sequence, window, modality, scale_name)), axis=1)
        result = pd.concat((result, self.calculate_profile(protein_sequence, prof_type, window, scale_name)), axis=1)
        result = pd.concat((result, self.calculate_arc(protein_sequence, modality, scale_name_arc)), axis=1)
        return result

    def get_all(self, protein_sequence: str, ph: float = 7, amide: bool = False, lamda_paac: int = 10,
                weight_paac: float = 0.05, lamda_apaac: int = 10, weight_apaac: float = 0.05, maxlag_socn: int = 45,
                maxlag_qso: int = 30, weight_qso: float = 0.1, window: int = 7,
                scalename: str = 'Eisenberg', scalename_arc: str = 'peparc', angle: int = 100,
                modality: str = 'max',
                prof_type: str = 'uH', tricomp: bool = False):

        """
        Calculate all descriptors from pydpi_py3 except tri-peptide pydpi_py3 and binary profiles

        Parameters
        ----------
        protein_sequence: str
            protein sequence

        ph: float (default 7.4)
            ph to calculate charge

        amide: bool (default True)
            by default is not considered an amide protein sequence

        lamda_paac: int
            parameter for PAAC default 10

        weight_paac: float
            parameter for PAAC default 0.05

        lamda_apaac: int
            parameter for APAAC default 10

        weight_apaac: float
            parameter for APAAC default 0.05

        maxlag_socn: int
            parameter for SOCN default 45

        maxlag_qso: int
            parameter for QSO default 30

        weight_qso: float
            parameter for QSO default 0.1

        window: int
            size of sliding window used amino acid window. If the sequence is shorter than the window,
            the length of the sequence is taken.

        scale_name: str
            name of the amino acid scale (one in
            https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the
            descriptor values. By default Eisenberg.

        scale_name_arc: str
            name of the amino acid scale (one in
            https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor
            values only binary scales. By default peparc.

        angle: int
            angle in which to calculate the moment. 100 for alpha helices, 180 for beta sheets

        modality: str
            maximum (max), mean (mean) or both (all) hydrophobic moment

        prof_type: str
            prof_type of profile, ‘H’ for hydrophobicity or ‘uH’ for hydrophobic moment

        tricomp: bool
            true or false to calculate or not tri-peptide pydpi_py3

        Returns
        -------
        dataframe: pd.DataFrame
            Dataframe with all features (value is variable)  for each sequence in the dataset
        """
        result = self.get_length(protein_sequence)
        result = pd.concat((result, self.get_charge(protein_sequence, ph, amide)), axis=1)
        result = pd.concat((result, self.get_charge_density(protein_sequence, ph, amide)), axis=1)
        result = pd.concat((result, self.get_formula(protein_sequence, amide)), axis=1)
        result = pd.concat((result, self.get_bond(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_molecular_weight(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_gravy(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_aromacity(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_isoelectric_point(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_instability_index(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_secondary_structure(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_molar_extinction_coefficient(protein_sequence)), axis=1)

        result = pd.concat((result, self.get_aliphatic_index(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_boman_index(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_hydrophobic_ratio(protein_sequence)), axis=1)

        # pydpi_base
        result = pd.concat((result, self.get_aa_comp(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_dp_comp(protein_sequence)), axis=1)
        if tricomp == True: result = pd.concat((result, self.get_tp_comp(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_moreau_broto_auto(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_moran_auto(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_geary_auto(protein_sequence)), axis=1)

        result = pd.concat((result, self.get_ctd(protein_sequence)), axis=1)
        result = pd.concat((result, self.get_conjoint_triad(protein_sequence)), axis=1)

        result = pd.concat((result, self.get_paac(protein_sequence, lamda_paac, weight_paac)), axis=1)
        result = pd.concat((result, self.get_apaac(protein_sequence, lamda_apaac, weight_apaac)), axis=1)
        result = pd.concat((result, self.get_socn(protein_sequence, maxlag_socn)), axis=1)

        result = pd.concat((result, self.get_qso(protein_sequence, maxlag_qso, weight_qso)), axis=1)

        # base class
        result = pd.concat((result, self.calculate_autocorrelation(protein_sequence, window, scalename)), axis=1)
        result = pd.concat((result, self.calculate_crosscorrelation(protein_sequence, window, scalename)), axis=1)
        result = pd.concat((result, self.calculate_moment(protein_sequence, window, angle, modality, scalename)),
                           axis=1)
        result = pd.concat((result, self.calculate_global(protein_sequence, window, modality, scalename)), axis=1)
        result = pd.concat((result, self.calculate_profile(protein_sequence, prof_type, window, scalename)), axis=1)
        result = pd.concat((result, self.calculate_arc(protein_sequence, modality, scalename_arc)), axis=1)
        return result
