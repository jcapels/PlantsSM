from unittest import TestCase, skip

from plants_sm.pathway_prediction.retroformer_reactor import Retroformer

# @skip
class TestRetroformer(TestCase):

    def test_retroformer(self):
        reactor = Retroformer()
        compound = "COC1=CC=C(C=C1)C2=COC3=CC(=C(C=C3C2=O)OC)O"
        reactor.react([compound])
        self.assertEqual(len(reactor.solutions), 10)