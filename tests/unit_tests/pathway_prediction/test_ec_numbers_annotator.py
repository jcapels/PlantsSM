import os
from unittest import TestCase, skip

import pandas as pd

from plants_sm.pathway_prediction.ec_numbers_annotator import ProtBertECAnnotator
from plants_sm.pathway_prediction.ec_numbers_annotator_utils.predictions import predict_with_model, predict_with_model_from_fasta

from tests import TEST_DIR

@skip("Require model download")
class TestEcNumbersAnnotatorUtils(TestCase):

    def setUp(self):
        self.data_path = os.path.join(TEST_DIR, "data", "test_data.csv")
        self.fasta_data_path = os.path.join(TEST_DIR, "data", "test.fasta")

    def test_make_prediction_protbert(self):
        results = predict_with_model(pipeline="DNN ProtBERT all data",
                           dataset_path=self.data_path,
                           output_path="predictions_protbert.csv",
                           ids_field="id",
                           sequences_field="sequence",
                           device="cpu")
        
        self.assertEqual(type(results), pd.DataFrame)

    def test_make_prediction_protbert_from_fasta(self):
        results = predict_with_model_from_fasta(pipeline="DNN ProtBERT all data",
                           fasta_path=self.fasta_data_path,
                           output_path="predictions_protbert_fasta.csv",
                           device="cpu")

        self.assertEqual(type(results), pd.DataFrame)

@skip("Require model download")
class TestEcNumbersAnnotator(TestCase):
    def setUp(self):
        self.annotator = ProtBertECAnnotator()  # Replace with actual annotator instance if needed
        self.data_path = os.path.join(TEST_DIR, "data", "test_data.csv")
        self.fasta_data_path = os.path.join(TEST_DIR, "data", "test.fasta")

    def test_annotate_from_fasta(self):
        # Assuming the annotator has a method to annotate from fasta
        solution = self.annotator.annotate_from_fasta(self.fasta_data_path)
        self.assertIsNotNone(solution)
        self.assertTrue(hasattr(solution, 'enzymes_ec_1'))
        self.assertTrue(hasattr(solution, 'enzymes_ec_2'))
        self.assertTrue(hasattr(solution, 'enzymes_ec_3'))
        self.assertTrue(hasattr(solution, 'enzymes_ec_4'))

        self.assertTrue(hasattr(solution, 'enzymes'))

        solution.get_score("P12345", "EC1")

    def test_annotate_from_csv(self):
        solution = self.annotator.annotate_from_csv(self.data_path)
        self.assertIsNotNone(solution)
        self.assertTrue(hasattr(solution, 'enzymes_ec_1'))
        self.assertTrue(hasattr(solution, 'enzymes_ec_2'))
        self.assertTrue(hasattr(solution, 'enzymes_ec_3'))
        self.assertTrue(hasattr(solution, 'enzymes_ec_4'))
        self.assertTrue(hasattr(solution, 'enzymes'))
