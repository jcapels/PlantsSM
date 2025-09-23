import os
from unittest import TestCase, skip

import pandas as pd

from plants_sm.pathway_prediction.ec_numbers_annotator import ESM1bECAnnotator, ESM2ECAnnotator, ProtBertECAnnotator

from plants_sm.pathway_prediction.ec_numbers_annotator_utils.esm1b_predictions import predict_with_esm1b_from_csv, predict_with_esm1b_from_fasta
from plants_sm.pathway_prediction.ec_numbers_annotator_utils.prot_bert_prediction import predict_with_protbert_from_csv, predict_with_protbert_from_fasta
from tests import TEST_DIR

@skip("Require model download")
class TestEcNumbersAnnotatorUtils(TestCase):

    def setUp(self):
        self.data_path = os.path.join(TEST_DIR, "data", "test_data.csv")
        self.fasta_data_path = os.path.join(TEST_DIR, "data", "test.fasta")

    def test_make_prediction_protbert(self):
        results = predict_with_protbert_from_csv(
                           dataset_path=self.data_path,
                           output_path="predictions_protbert.csv",
                           ids_field="id",
                           sequences_field="sequence",
                           device="cpu")
        
        self.assertEqual(type(results), pd.DataFrame)

    def test_make_prediction_protbert_from_fasta(self):
        results = predict_with_protbert_from_fasta(
                           fasta_path=self.fasta_data_path,
                           output_path="predictions_protbert_fasta.csv",
                           device="cpu")

        self.assertEqual(type(results), pd.DataFrame)

    def test_make_prediction_esm1b(self):
        results = predict_with_esm1b_from_csv(
                           dataset_path=self.data_path,
                           output_path="predictions_esm1b.csv",
                           ids_field="id",
                           sequences_field="sequence",
                           device="cpu")
        
        self.assertEqual(type(results), pd.DataFrame)

    def test_make_prediction_esm1b_from_fasta(self):
        results = predict_with_esm1b_from_fasta(
                           fasta_path=self.fasta_data_path,
                           output_path="predictions_esm1b_fasta.csv",
                           device="cpu")

        self.assertEqual(type(results), pd.DataFrame)

@skip("Require model download")
class TestProtBertEcNumbersAnnotator(TestCase):
    def setUp(self):
        self.annotator = ProtBertECAnnotator()  # Replace with actual annotator instance if needed
        self.data_path = os.path.join(TEST_DIR, "data", "test_data.csv")
        self.fasta_data_path = os.path.join(TEST_DIR, "data", "test.fasta")

    def test_annotate_from_fasta(self):
        # Assuming the annotator has a method to annotate from fasta
        solution = self.annotator.annotate_from_file(self.fasta_data_path, "fasta")
        self.assertIsNotNone(solution)
        self.assertTrue(hasattr(solution, 'entity_ec_1'))
        self.assertTrue(hasattr(solution, 'entity_ec_2'))
        self.assertTrue(hasattr(solution, 'entity_ec_3'))
        self.assertTrue(hasattr(solution, 'entity_ec_4'))

        self.assertTrue(hasattr(solution, 'entities'))

        self.assertEqual(solution.get_score("Q7XQ85", "EC1")[0][0], "4")

    def test_annotate_from_fasta_wgpu(self):
        # Assuming the annotator has a method to annotate from fasta
        solution = self.annotator.annotate_from_file(self.fasta_data_path, "fasta", device="cuda:2")
        self.assertIsNotNone(solution)
        self.assertTrue(hasattr(solution, 'entity_ec_1'))
        self.assertTrue(hasattr(solution, 'entity_ec_2'))
        self.assertTrue(hasattr(solution, 'entity_ec_3'))
        self.assertTrue(hasattr(solution, 'entity_ec_4'))

        self.assertTrue(hasattr(solution, 'entities'))

        self.assertEqual(solution.get_score("A0A0P0WIY3", "EC1")[0][0], "4")

    def test_annotate_from_csv(self):
        solution = self.annotator.annotate_from_file(self.data_path, "csv")
        self.assertIsNotNone(solution)
        self.assertTrue(hasattr(solution, 'entity_ec_1'))
        self.assertTrue(hasattr(solution, 'entity_ec_2'))
        self.assertTrue(hasattr(solution, 'entity_ec_3'))
        self.assertTrue(hasattr(solution, 'entity_ec_4'))
        self.assertTrue(hasattr(solution, 'entities'))

        self.assertEqual(solution.get_score("O16025", "EC1")[0][0], "1")

@skip("Require model download")
class TestESM1bEcNumbersAnnotator(TestCase):
    def setUp(self):
        self.annotator = ESM1bECAnnotator()  # Replace with actual annotator instance if needed
        self.data_path = os.path.join(TEST_DIR, "data", "test_data.csv")
        self.fasta_data_path = os.path.join(TEST_DIR, "data", "test.fasta")

    def test_annotate_from_fasta(self):
        # Assuming the annotator has a method to annotate from fasta
        solution = self.annotator.annotate_from_file(self.fasta_data_path, "fasta")
        self.assertIsNotNone(solution)
        self.assertTrue(hasattr(solution, 'entity_ec_1'))
        self.assertTrue(hasattr(solution, 'entity_ec_2'))
        self.assertTrue(hasattr(solution, 'entity_ec_3'))
        self.assertTrue(hasattr(solution, 'entity_ec_4'))

        self.assertTrue(hasattr(solution, 'entities'))

        self.assertEqual(solution.get_score("Q7XQ85", "EC1")[0][0], "4")

    def test_annotate_from_fasta_wgpu(self):
        # Assuming the annotator has a method to annotate from fasta
        solution = self.annotator.annotate_from_file(self.fasta_data_path, "fasta", device="cuda:2")
        self.assertIsNotNone(solution)
        self.assertTrue(hasattr(solution, 'entity_ec_1'))
        self.assertTrue(hasattr(solution, 'entity_ec_2'))
        self.assertTrue(hasattr(solution, 'entity_ec_3'))
        self.assertTrue(hasattr(solution, 'entity_ec_4'))

        self.assertTrue(hasattr(solution, 'entities'))

        self.assertEqual(solution.get_score("A0A0P0WIY3", "EC1")[0][0], "4")

    def test_annotate_from_csv(self):
        solution = self.annotator.annotate_from_file(self.data_path, "csv")
        self.assertIsNotNone(solution)
        self.assertTrue(hasattr(solution, 'entity_ec_1'))
        self.assertTrue(hasattr(solution, 'entity_ec_2'))
        self.assertTrue(hasattr(solution, 'entity_ec_3'))
        self.assertTrue(hasattr(solution, 'entity_ec_4'))
        self.assertTrue(hasattr(solution, 'entities'))

        self.assertEqual(solution.get_score("O16025", "EC1")[0][0], "1")

@skip("Require model download")
class TestESM2EcNumbersAnnotator(TestCase):
    def setUp(self):
        self.annotator = ESM2ECAnnotator()  # Replace with actual annotator instance if needed
        self.data_path = os.path.join(TEST_DIR, "data", "test_data.csv")
        self.fasta_data_path = os.path.join(TEST_DIR, "data", "test.fasta")

    def test_annotate_from_fasta(self):
        # Assuming the annotator has a method to annotate from fasta
        solution = self.annotator.annotate_from_file(self.fasta_data_path, "fasta")
        self.assertIsNotNone(solution)
        self.assertTrue(hasattr(solution, 'entity_ec_1'))
        self.assertTrue(hasattr(solution, 'entity_ec_2'))
        self.assertTrue(hasattr(solution, 'entity_ec_3'))
        self.assertTrue(hasattr(solution, 'entity_ec_4'))

        self.assertTrue(hasattr(solution, 'entities'))

        self.assertEqual(solution.get_score("Q7XQ85", "EC1")[0][0], "4")

    def test_annotate_from_fasta_wgpu(self):
        # Assuming the annotator has a method to annotate from fasta
        solution = self.annotator.annotate_from_file(self.fasta_data_path, "fasta", device="cuda:2", num_gpus=4)
        self.assertIsNotNone(solution)
        self.assertTrue(hasattr(solution, 'entity_ec_1'))
        self.assertTrue(hasattr(solution, 'entity_ec_2'))
        self.assertTrue(hasattr(solution, 'entity_ec_3'))
        self.assertTrue(hasattr(solution, 'entity_ec_4'))

        self.assertTrue(hasattr(solution, 'entities'))

        self.assertEqual(solution.get_score("A0A0P0WIY3", "EC1")[0][0], "4")

    def test_annotate_from_csv(self):
        solution = self.annotator.annotate_from_file(self.data_path, "csv")
        self.assertIsNotNone(solution)
        self.assertTrue(hasattr(solution, 'entity_ec_1'))
        self.assertTrue(hasattr(solution, 'entity_ec_2'))
        self.assertTrue(hasattr(solution, 'entity_ec_3'))
        self.assertTrue(hasattr(solution, 'entity_ec_4'))
        self.assertTrue(hasattr(solution, 'entities'))

        self.assertEqual(solution.get_score("O16025", "EC1")[0][0], "1")

