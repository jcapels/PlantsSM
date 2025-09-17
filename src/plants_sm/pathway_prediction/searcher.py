from abc import ABC, abstractmethod
from typing import List, Union

from plants_sm.pathway_prediction.entities import Molecule
from plants_sm.pathway_prediction.reactor import Reactor
from plants_sm.pathway_prediction.solution import ReactionSolution, Solution

class SearcherSolution(Solution):

    route: List[List[Union[None, str]]]
    template: List[List[Union[None, str]]]
    success: List[bool]
    depth: List[int]
    counts: List[int]
    reaction_solutions: List[Union[None, ReactionSolution]]

    def to_csv(self, path):
        pass

class Searcher(ABC):

    reactors: List[Reactor]

    @abstractmethod
    def search(self, molecule: Molecule) -> SearcherSolution:
        pass

    @abstractmethod
    def run_reactor(self, molecule: Molecule) -> List[ReactionSolution]:
        pass