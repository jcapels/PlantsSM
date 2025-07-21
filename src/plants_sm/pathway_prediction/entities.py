from abc import ABC
from typing import List, Union
from pydantic import BaseModel
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
    
class Molecule(BiologicalEntity):
    
    smiles: str
    substrate: bool = None
    cofactor: bool = None
    mol: Mol = None

    class Config:
        """
        Model Configuration: https://pydantic-docs.helpmanual.io/usage/model_config/
        """
        extra = 'allow'
        allow_mutation = True
        validate_assignment = True
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

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

    class Config:
        """
        Model Configuration: https://pydantic-docs.helpmanual.io/usage/model_config/
        """
        extra = 'allow'
        allow_mutation = True
        validate_assignment = True
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

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
