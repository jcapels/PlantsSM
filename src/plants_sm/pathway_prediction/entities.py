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

    representation: Union[str, Representation]

    def __str__(self):
        return str(self.representation)
    
class Protein(BiologicalEntity):

    @property
    def sequence(self):
        return self.representation

    @classmethod
    def from_sequence(cls, sequence: str) -> 'ReactionSmarts':
        is_valid = SequenceUtils.is_valid_protein_sequence(sequence)
        if not is_valid:
            raise ValueError("Amino-acid sequence not valid")

        return cls(representation=sequence)
    
    @classmethod
    def from_fasta(cls, file) -> Dict[str, 'Protein']:
        from Bio import SeqIO

        proteins = {}
        # Read a FASTA file
        for record in SeqIO.parse(file, "fasta"):
            proteins[record.id] = cls.from_sequence(str(record.seq))
        return proteins
    
    @classmethod
    def from_csv(cls, file: str, **kwargs) -> Dict[str, 'Protein']:
        """
        Read a CSV file and create Protein objects from the sequences.
        The CSV file should have columns "id" and "sequence".
        Parameters
        ----------
        file : str
            Path to the CSV file.
        **kwargs : dict
            Additional keyword arguments to pass to pandas read_csv.
        Returns
        -------
        Dict[str, Protein]
            A dictionary mapping protein IDs to Protein objects.
        """

        proteins = {}
        # Read a CSV file
        df = pd.read_csv(file, **kwargs)
        for _, row in df.iterrows():
            proteins[row["id"]] = cls.from_sequence(row["sequence"])
            
        return proteins

    
class Molecule(BiologicalEntity):
    
    smiles: str
    substrate: bool = None
    cofactor: bool = None
    mol: Mol = None

    class Config:
        arbitrary_types_allowed = True

    @validator('mol', pre=True, always=True, allow_reuse=True)
    def validate_mol(cls, value):
        if not isinstance(value, Mol):
            raise ValueError("mol must be a RDKit Mol instance")
        return value

    @classmethod
    def from_smiles(cls, smiles: str) -> 'Molecule':
        smiles_repr = Smiles.from_smiles(smiles)
        mol = smiles_repr.to_mol()
        return cls(smiles = smiles, mol = mol, representation = smiles_repr)
    
    @classmethod
    def from_mol(cls, mol: Mol) -> 'Molecule':
        smiles = MolToSmiles(mol, canonical=True)
        smiles_repr = Smiles(mol=mol, representation = smiles)
        return cls(smiles = smiles, mol = mol, representation = smiles_repr)
    
class Reaction(BiologicalEntity):

    reactants: List[Molecule] = None
    products: List[Molecule] = None
    ec_numbers: List[str] = None
    reaction: ReactionSmarts
    score: float = None

    @property
    def smarts(self):
        return self.representation
    
    @property
    def smiles(self):
        return self.representation
    
    @classmethod
    def from_smarts(cls, smarts) -> 'Reaction':
        reaction = ReactionSmarts.from_smarts(smarts)

        return cls(reaction=reaction, representation=smarts)
    
    @classmethod
    def from_smiles(cls, smiles: str) -> 'Smiles':
        reactants = []
        products = []

        reaction = ReactionSmarts.from_smarts(smiles)
        reactants_set, products_set = ChemUtils.break_reaction(smiles)

        for reactant in reactants_set:
            reactants.append(Molecule.from_smiles(reactant))
        for product in products_set:
            products.append(Molecule.from_smiles(product))

        return cls(reaction=reaction, representation=smiles, products=products, reactants=reactants)
