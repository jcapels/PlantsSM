from abc import abstractmethod
from typing import List
from pydantic import BaseModel

from plants_sm.pathway_prediction.entities import Molecule
from plants_sm.pathway_prediction.reactor import Reactor
from plants_sm.pathway_prediction.solution import Solution

class SearcherSolution(Solution):
    pass


class Searcher(BaseModel):

    reactors: List[Reactor]

    @abstractmethod
    def search(self, molecule: Molecule) -> SearcherSolution:
        pass