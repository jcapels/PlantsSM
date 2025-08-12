from unittest import TestCase

from plants_sm.pathway_prediction.reaction_rules_reactor import ReactionRulesReactor


class TestReactionRulesReactor(TestCase):

    def test_react(self):
        reactor = ReactionRulesReactor(score_threshold=10000)
        compound = "COC1=CC=C(C=C1)C2=COC3=CC(=C(C=C3C2=O)OC)O"
        reactor.react([compound])
        self.assertEqual(len(reactor.solutions), 98)