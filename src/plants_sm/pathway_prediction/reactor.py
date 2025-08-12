from abc import ABC, abstractmethod
from typing import List, Union
from pydantic import BaseModel

from plants_sm.pathway_prediction.entities import Molecule
from plants_sm.pathway_prediction.solution import ReactionSolution
from rdkit.Chem import Mol

class Reactor(ABC):

    solutions: List[ReactionSolution] = None

    def react(self, reactants: Union[str, Mol]) -> List[ReactionSolution]:
        
        if isinstance(reactants[0], str):
            for i in range(len(reactants)):
                reactants[i] = Molecule.from_smiles(reactants[i]).mol

        self.solutions = self._react(reactants)
        return self.solutions

    @abstractmethod
    def _react(self, reactants: List[Mol]) -> List[ReactionSolution]:
        pass
