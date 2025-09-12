from abc import ABC
from typing import Dict, List, Union
import pandas as pd
from pydantic import BaseModel, validator
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import MolToSmiles
from plants_sm.pathway_prediction._chem_utils import ChemUtils
from plants_sm.pathway_prediction._sequence_utils import SequenceUtils
from plants_sm.pathway_prediction.representations import ReactionSmarts, Representation, Smiles

class BiologicalEntity(BaseModel):
    """
    Base class for biological entities, such as proteins and molecules.

    Attributes
    ----------
    representation : Union[str, Representation]
        The representation of the biological entity.
    """

    representation: Union[str, Representation]

    def __str__(self):
        """
        Return the string representation of the biological entity.

        Returns
        -------
        str
            The string representation of the entity.
        """
        return str(self.representation)

class Protein(BiologicalEntity):
    """
    Class representing a protein, derived from BiologicalEntity.

    Attributes
    ----------
    representation : str
        The amino acid sequence of the protein.
    """

    @property
    def sequence(self) -> str:
        """
        Get the amino acid sequence of the protein.

        Returns
        -------
        str
            The amino acid sequence.
        """
        return self.representation

    @classmethod
    def from_sequence(cls, sequence: str) -> 'Protein':
        """
        Create a Protein object from an amino acid sequence.

        Parameters
        ----------
        sequence : str
            The amino acid sequence.

        Returns
        -------
        Protein
            A Protein object.

        Raises
        ------
        ValueError
            If the sequence is not a valid protein sequence.
        """
        is_valid = SequenceUtils.is_valid_protein_sequence(sequence)
        if not is_valid:
            raise ValueError("Amino-acid sequence not valid")
        return cls(representation=sequence)

    @classmethod
    def from_sequences(cls, ids: List[Union[str, int]], sequences: List[str]) -> Dict[str, 'Protein']:
        """
        Create a dictionary of Protein objects from lists of IDs and sequences.

        Parameters
        ----------
        ids : List[Union[str, int]]
            List of protein IDs.
        sequences : List[str]
            List of amino acid sequences.

        Returns
        -------
        Dict[str, Protein]
            A dictionary mapping protein IDs to Protein objects.
        """
        proteins = {}
        for i, id_ in enumerate(ids):
            proteins[id_] = cls.from_sequence(sequences[i])
        return proteins

    @classmethod
    def from_fasta(cls, file) -> Dict[str, 'Protein']:
        """
        Create a dictionary of Protein objects from a FASTA file.

        Parameters
        ----------
        file : str
            Path to the FASTA file.

        Returns
        -------
        Dict[str, Protein]
            A dictionary mapping protein IDs to Protein objects.
        """
        from Bio import SeqIO
        proteins = {}
        for record in SeqIO.parse(file, "fasta"):
            proteins[record.id] = cls.from_sequence(str(record.seq))
        return proteins

    @classmethod
    def from_csv(cls, file: str, **kwargs) -> Dict[str, 'Protein']:
        """
        Create a dictionary of Protein objects from a CSV file.

        The CSV file should have columns "id" and "sequence".

        Parameters
        ----------
        file : str
            Path to the CSV file.
        **kwargs : dict
            Additional keyword arguments to pass to pandas.read_csv.

        Returns
        -------
        Dict[str, Protein]
            A dictionary mapping protein IDs to Protein objects.
        """
        proteins = {}
        df = pd.read_csv(file, **kwargs)
        for _, row in df.iterrows():
            proteins[row["id"]] = cls.from_sequence(row["sequence"])
        return proteins

class Molecule(BiologicalEntity):
    """
    Class representing a molecule, derived from BiologicalEntity.

    Attributes
    ----------
    smiles : str
        The SMILES string of the molecule.
    substrate : bool, optional
        Whether the molecule is a substrate.
    cofactor : bool, optional
        Whether the molecule is a cofactor.
    mol : Mol
        The RDKit Mol object of the molecule.
    """

    smiles: str
    substrate: bool = None
    cofactor: bool = None
    mol: Mol = None

    class Config:
        arbitrary_types_allowed = True

    @validator('mol', pre=True, always=True, allow_reuse=True)
    def validate_mol(cls, value):
        """
        Validate that the 'mol' attribute is an RDKit Mol instance.

        Parameters
        ----------
        value : Any
            The value to validate.

        Returns
        -------
        Mol
            The validated RDKit Mol object.

        Raises
        ------
        ValueError
            If the value is not an RDKit Mol instance.
        """
        if not isinstance(value, Mol):
            raise ValueError("mol must be a RDKit Mol instance")
        return value

    @classmethod
    def from_smiles(cls, smiles: str) -> 'Molecule':
        """
        Create a Molecule object from a SMILES string.

        Parameters
        ----------
        smiles : str
            The SMILES string.

        Returns
        -------
        Molecule
            A Molecule object.
        """
        smiles_repr = Smiles.from_smiles(smiles)
        mol = smiles_repr.to_mol()
        return cls(smiles=smiles, mol=mol, representation=smiles_repr)

    @classmethod
    def from_mol(cls, mol: Mol) -> 'Molecule':
        """
        Create a Molecule object from an RDKit Mol object.

        Parameters
        ----------
        mol : Mol
            The RDKit Mol object.

        Returns
        -------
        Molecule
            A Molecule object.
        """
        smiles = MolToSmiles(mol, canonical=True)
        smiles_repr = Smiles(mol=mol, representation=smiles)
        return cls(smiles=smiles, mol=mol, representation=smiles_repr)

class Reaction(BiologicalEntity):
    """
    Class representing a chemical reaction, derived from BiologicalEntity.

    Attributes
    ----------
    reactants : List[Molecule], optional
        List of reactant molecules.
    products : List[Molecule], optional
        List of product molecules.
    ec_numbers : List[str], optional
        List of EC numbers associated with the reaction.
    reaction : ReactionSmarts
        The ReactionSmarts object representing the reaction.
    score : float, optional
        The score of the reaction.
    """

    reactants: List[Molecule] = None
    products: List[Molecule] = None
    ec_numbers: List[str] = None
    reaction: ReactionSmarts
    score: float = None

    @property
    def smarts(self) -> str:
        """
        Get the SMARTS string of the reaction.

        Returns
        -------
        str
            The SMARTS string.
        """
        return self.representation

    @property
    def smiles(self) -> str:
        """
        Get the SMILES string of the reaction.

        Returns
        -------
        str
            The SMILES string.
        """
        return self.representation

    @classmethod
    def from_smarts(cls, smarts: str) -> 'Reaction':
        """
        Create a Reaction object from a SMARTS string.

        Parameters
        ----------
        smarts : str
            The SMARTS string.

        Returns
        -------
        Reaction
            A Reaction object.
        """
        reaction = ReactionSmarts.from_smarts(smarts)
        return cls(reaction=reaction, representation=smarts)

    @classmethod
    def from_smiles(cls, smiles: str) -> 'Reaction':
        """
        Create a Reaction object from a reaction SMILES string.

        Parameters
        ----------
        smiles : str
            The reaction SMILES string.

        Returns
        -------
        Reaction
            A Reaction object.
        """
        reactants = []
        products = []
        reaction = ReactionSmarts.from_smarts(smiles)
        reactants_set, products_set = ChemUtils.break_reaction(smiles)
        for reactant in reactants_set:
            reactants.append(Molecule.from_smiles(reactant))
        for product in products_set:
            products.append(Molecule.from_smiles(product))
        return cls(reaction=reaction, representation=smiles, products=products, reactants=reactants)
