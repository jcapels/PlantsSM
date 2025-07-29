from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

from plants_sm.pathway_prediction.entities import Molecule
from plants_sm.pathway_prediction.reactor import Reactor
from plants_sm.pathway_prediction.solution import ReactionSolution, Solution

class SearcherSolution(Solution):
    pass


class Searcher(ABC):

    reactors: List[Reactor]

    @abstractmethod
    def search(self, molecule: Molecule) -> SearcherSolution:
        pass

    @abstractmethod
    def run_reactor(self, molecule: Molecule) -> List[ReactionSolution]:
        pass