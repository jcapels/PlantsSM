
from abc import abstractmethod
from typing import Any

from pydantic import BaseModel
    
from plants_sm.pathway_prediction._chem_utils import ChemUtils
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts

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
        """
        Model Configuration: https://pydantic-docs.helpmanual.io/usage/model_config/
        """
        extra = 'allow'
        allow_mutation = True
        validate_assignment = True
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

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
        """
        Model Configuration: https://pydantic-docs.helpmanual.io/usage/model_config/
        """
        extra = 'allow'
        allow_mutation = True
        validate_assignment = True
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

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