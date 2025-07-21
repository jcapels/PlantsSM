import os
from unittest import TestCase

import pandas as pd

from plants_sm.pathway_prediction.ec_numbers_annotator_utils.predictions import predict_with_model

from tests import TEST_DIR

class TestEcNumbersAnnotatorUtils(TestCase):

    def setUp(self):
        self.data_path = os.path.join(TEST_DIR, "data", "test_data.csv")

    def test_make_prediction_protbert(self):
        results = predict_with_model(pipeline="DNN ProtBERT all data",
                           dataset_path=self.data_path,
                           output_path="predictions_protbert.csv",
                           ids_field="id",
                           sequences_field="sequence",
                           device="cuda:0")
        
        self.assertEqual(type(results), pd.DataFrame)