from unittest import TestCase, skip

from plants_sm.pathway_prediction.MCTS_A import MCTS_A
from plants_sm.pathway_prediction.entities import Molecule
from plants_sm.pathway_prediction.retroformer_reactor import Retroformer
from plants_sm.pathway_prediction.solution import ReactionSolution

@skip("For now")
class TestMCTS_A(TestCase):

    def test_search(self):
        searcher = MCTS_A(reactors=[Retroformer()], device='cuda:1', simulations=100, cpuct=4.0, times=3000)
        solution = searcher.search(molecule=Molecule.from_smiles("COC1=CC=C(C=C1)C2=COC3=CC(=C(C=C3C2=O)OC)O"))
        self.assertIsInstance(solution, ReactionSolution)