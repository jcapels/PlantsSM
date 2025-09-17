

from typing import List
from plants_sm.pathway_prediction.proteome_compass import ProteomeCompass
from plants_sm.pathway_prediction.reactor import Reactor

from rdkit.Chem import Mol

from plants_sm.pathway_prediction.solution import ReactionSolution


class ProteomeCompassedReactor(Reactor):

    def __init__(self, proteome_compass: ProteomeCompass, reactor: Reactor):

        self.proteome_compass = proteome_compass
        self.reactor = reactor
    
    def _react(self, reactants: List[Mol]) -> List[ReactionSolution]:
        
        solutions = self.reactor.react(reactants)

        solutions = self.proteome_compass.direct(solutions)

        return solutions
