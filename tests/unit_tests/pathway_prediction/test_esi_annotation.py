import os
from unittest import TestCase, skip

import pandas as pd

from plants_sm.pathway_prediction.esi_annotator import ProtBertESIAnnotator
from tests import TEST_DIR

class TestESIAnnotator(TestCase):

    def setUp(self):
        
        self.data_path = os.path.join(TEST_DIR,
                                      "data", 
                                      "multi_input_dataset.csv")
        self.annotator = ProtBertESIAnnotator()
        
    def test_esi_annotator(self):
        solution = self.annotator.annotate_from_csv(self.data_path)