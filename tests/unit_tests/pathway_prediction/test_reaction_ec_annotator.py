import os
from unittest import TestCase

import pandas as pd

from plants_sm.pathway_prediction.reaction_ec_number_annotator import ReactionECNumberAnnotator
from tests import TEST_DIR


class TestEcNumbersAnnotatorReactions(TestCase):

    def setUp(self):
        self.test_csv = os.path.join(TEST_DIR, "data", "reaction_ec_data.csv")

    def test_annotate_reactions(self):
        annotator = ReactionECNumberAnnotator()

        entities = pd.read_csv(self.test_csv)

        solution = annotator.annotate(entities)

        self.assertEqual(len(solution.entity_ec_3), entities.shape[0] - 1)