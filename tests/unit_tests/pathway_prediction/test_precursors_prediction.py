
from unittest import TestCase

from plants_sm.pathway_prediction.precursor_prediction import predict_precursors, export_precursors


class TestPrecursorPrediction(TestCase):

    def test_predict_precursors(self):
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "C1=CC=C(C=C1)C=O"]
        predictions = predict_precursors(smiles)
        self.assertEqual(len(predictions), 2)
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(predictions[0], str)
        self.assertIn("", predictions[0])
        self.assertIn("Phenylalanine", predictions[1])
        print(predictions)

    def test_export_precursors(self):
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "C1=CC=C(C=C1)C=O"]
        result = export_precursors(smiles)
        self.assertTrue(result)