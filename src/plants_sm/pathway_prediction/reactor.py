from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

from plants_sm.pathway_prediction.solution import ReactionSolution

class Reactor(ABC):

    solutions: List[ReactionSolution] = None

    def react(self, reactants: List[str]):

        self.solutions = self._react(reactants)
        return self.solutions

    @abstractmethod
    def _react(self, reactants: List[str]) -> List[ReactionSolution]:
        pass
