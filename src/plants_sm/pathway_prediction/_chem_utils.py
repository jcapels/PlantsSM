from typing import Union, List

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MolFromSmiles, Mol, MolToSmiles, RemoveHs, AllChem, Descriptors
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from rdkit.Chem.rdChemReactions import ReactionFromSmarts, ChemicalReaction



class ChemUtils:
    """
    A class containing utility functions using RDKit.
    """

    @staticmethod
    def smiles_to_isomerical_smiles(smiles: str):
        """
        Converts a molecule to its canonical SMILES.

        Parameters
        ----------
        smiles: str
            The SMILES of the molecule.

        Returns
        -------
        str
            The SMILES string.
        """
        try:
            return MolToSmiles(RemoveHs(MolFromSmiles(smiles)), isomericSmiles=True)
        except TypeError:
            return None

    @staticmethod
    def smiles_to_mol(smiles: str):
        """
        Converts a SMILES string to an RDKit molecule.

        Parameters
        ----------
        smiles: str
            The SMILES string.

        Returns
        -------
        Mol
            The RDKit molecule.
        """
        try:
            mol = MolFromSmiles(smiles)
            return mol
        except :
            return None
        
    @staticmethod
    def mol_to_cannonical_smiles(mol: Mol):
        try:
            smiles = MolToSmiles(mol)
            return smiles
        except :
            return None


    @staticmethod
    def rdkit_logs(enable=False):
        if not enable:
            RDLogger.DisableLog('rdApp.*')

    @staticmethod
    def validate_smiles(smiles: str):
        """
        Validates one SMILES.
        Returns True if SMILES is valid.

        Parameters
        ----------
        smiles: str
            The SMILES to validate.

        Returns
        -------
        List[str]
            The valid SMILES.
        """
        if MolFromSmiles(smiles) is None:
            return False
        else:
            return True

    @staticmethod
    def _smarts_to_reaction(reaction_smarts: str):
        """
        Converts a SMARTS string to a ChemicalReaction object.

        Parameters
        ----------
        reaction_smarts: str
            The SMARTS string.

        Returns
        -------
        ChemicalReaction
            The reaction.
        """
        try:
            return ReactionFromSmarts(reaction_smarts)
        except ValueError:
            return None
        
    @staticmethod
    def break_reaction(smiles: str):
        """
        
        """
        reactants, products = smiles.split(">>")
        reactants_set = reactants.split(".")
        products_set = products.split(".")
        return reactants_set, products_set
        
    @staticmethod
    def _remove_hs(mol: Mol):
        """
        Removes hydrogen atoms from a molecule.

        Parameters
        ----------
        mol: Mol
            The molecule to remove hydrogen atoms from.

        Returns
        -------
        Mol
            The molecule wit implicit hydrogen atoms.
        """
        if mol is None:
            return None
        try:
            return RemoveHs(mol)
        except Chem.rdchem.KekulizeException:
            return mol
        except Chem.rdchem.AtomValenceException:
            return mol
        except Chem.rdchem.AtomKekulizeException:
            return mol

    @staticmethod
    def _sanitize_mol(mol: Mol):
        """
        Sanitizes a molecule.

        Parameters
        ----------
        mol: Mol
            The molecule to sanitize.

        Returns
        -------
        Mol
            The sanitized molecule.
        """
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
            return mol
        except ValueError:
            return None

    @staticmethod
    def react(smiles: Union[str, List[str]], smarts: str):
        """
        Reacts a molecule with a reaction.

        Parameters
        ----------
        smiles: Union[str, List[str]]
            The smiles of the reactant(s)' molecule(s).
        smarts: str
            The SMARTS string of the reaction.

        Returns
        -------
        list of str
            The list of products.
        """
        if isinstance(smiles, str):
            mol = (MolFromSmiles(smiles),)
        else:
            mol = [MolFromSmiles(s) for s in smiles]
        reaction = ChemUtils._smarts_to_reaction(smarts)
        res = []
        try:
            ps = reaction.RunReactants(mol)
            for pset in ps:
                pset = [MolToSmiles(ChemUtils._sanitize_mol(pset_i)) for pset_i in pset]
                if None not in pset:
                    res.append(pset)
        except:
            return []
        
        return res
        

    @staticmethod
    def _create_reaction_instances(rxn: ChemicalReaction, reactants: List[Mol]):
        """
        Creates reaction smiles from a reaction and a list of reactants.

        Parameters
        ----------
        rxn: ChemicalReaction
            The reaction.
        reactants: List[Mol]
            The reactants.

        Returns
        -------
        list of str
            The list of reaction smiles.
        """
        res = []
        ps = rxn.RunReactants(reactants)
        for pset in ps:
            pset = [ChemUtils._sanitize_mol(pset_i) for pset_i in pset]
            if None not in pset:
                tres = ChemicalReaction()
                for p in pset:
                    tres.AddProductTemplate(ChemUtils._remove_hs(p))
                for reactant in reactants:
                    tres.AddReactantTemplate(ChemUtils._remove_hs(reactant))
                res.append(tres)
        return list(set([AllChem.ReactionToSmiles(entry, canonical=True) for entry in res]))

    @staticmethod
    def uncharge_smiles(smiles: str):
        """
        Neutralizes a molecule.

        Parameters
        ----------
        smiles: str
            The molecule smiles to uncharge.
        Returns
        -------
        str
            The uncharged molecule smiles.
        """
        mol = MolFromSmiles(smiles)
        if mol:
            uncharger = Uncharger()
            return MolToSmiles(uncharger.uncharge(mol))
        return smiles

