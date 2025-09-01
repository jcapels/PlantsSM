
from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, validator
    
from plants_sm.pathway_prediction._chem_utils import ChemUtils
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction

class Representation(BaseModel):

    representation: Any

    @property
    @abstractmethod
    def _str_representation(self):
        pass

    def __str__(self):
        return self._str_representation

class Smiles(Representation):

    mol: Mol

    class Config:
        arbitrary_types_allowed = True

    @validator('mol', pre=True, always=True, allow_reuse=True)
    def validate_mol(cls, value):
        if not isinstance(value, Mol):
            raise ValueError("mol must be a RDKit Mol instance")
        return value

    @property
    def _str_representation(self):
        return self.representation
    
    
    @classmethod
    def from_smiles(cls, smiles: str) -> 'Smiles':
        mol = ChemUtils.smiles_to_mol(smiles)
        if mol is not None:
            return cls(representation = smiles, mol = mol)
        else:
            raise ValueError("SMILES is not valid")
    
    def to_mol(self) -> Mol:
        return self.mol
    
class ReactionSmarts(Representation):

    reaction: ChemicalReaction

    class Config:
        arbitrary_types_allowed = True

    @validator('reaction', pre=True, always=True, allow_reuse=True)
    def validate_reaction(cls, value):
        if not isinstance(value, ChemicalReaction):
            raise ValueError("reaction must be a ChemicalReaction instance")
        return value

    @property
    def _str_representation(self):
        return self.representation
    
    @classmethod
    def from_smarts(cls, smarts: str) -> 'ReactionSmarts':
        reaction = ChemUtils._smarts_to_reaction(smarts)
        if reaction is None:
            raise ValueError("SMARTS not valid")

        return cls(reaction=reaction, representation=smarts)
    
    def to_chemical_reaction(self):
        return self.reaction