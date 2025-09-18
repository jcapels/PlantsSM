from typing import List
from plants_sm.pathway_prediction.proteome_compass import ProteomeCompass
from plants_sm.pathway_prediction.reactor import Reactor
from rdkit.Chem import Mol
from plants_sm.pathway_prediction.solution import ReactionSolution

class ProteomeCompassedReactor(Reactor):
    """A reactor that uses a ProteomeCompass to score and annotate reaction solutions.

    This class wraps a standard Reactor and applies a ProteomeCompass to the reaction solutions
    produced by the reactor, enhancing them with enzyme and substrate annotations and scores.

    Attributes
    ----------
    proteome_compass : ProteomeCompass
        The ProteomeCompass instance used to score and annotate reaction solutions.
    reactor : Reactor
        The underlying reactor used to generate reaction solutions.
    """

    def __init__(self, proteome_compass: ProteomeCompass, reactor: Reactor):
        """Initialize the ProteomeCompassedReactor with a ProteomeCompass and a Reactor.

        Parameters
        ----------
        proteome_compass : ProteomeCompass
            The ProteomeCompass instance used to score and annotate reaction solutions.
        reactor : Reactor
            The underlying reactor used to generate reaction solutions.
        """
        self.proteome_compass = proteome_compass
        self.reactor = reactor

    def _react(self, reactants: List[Mol]) -> List[ReactionSolution]:
        """Generate and score reaction solutions for a list of reactants.

        This method first generates reaction solutions using the underlying reactor,
        then applies the ProteomeCompass to score and annotate the solutions.

        Parameters
        ----------
        reactants : List[Mol]
            List of RDKit Mol objects representing the reactants.

        Returns
        -------
        List[ReactionSolution]
            List of reaction solutions, scored and annotated by the ProteomeCompass.
        """
        solutions = self.reactor.react(reactants)
        solutions = self.proteome_compass.direct(solutions)
        return solutions
