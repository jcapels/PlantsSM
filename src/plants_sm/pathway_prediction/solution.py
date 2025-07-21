
from pydantic import BaseModel
from typing import Dict, List

from plants_sm.pathway_prediction.entities import Molecule, Protein, Reaction

class Solution(BaseModel):

    score : float = None

    class Config:
        """
        Model Configuration: https://pydantic-docs.helpmanual.io/usage/model_config/
        """
        extra = 'allow'
        allow_mutation = True
        validate_assignment = True
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True


class ReactionSolution(Solution):

    reactants: List[Molecule]
    products: List[Molecule]
    reaction: Reaction
    ec_numbers: List[str] = None

class ECSolution(Solution):

    enzymes_ec = Dict[str, List[str]]
    enzymes = Dict[str, Protein] # dictionary to trace the sequence by the ID
    