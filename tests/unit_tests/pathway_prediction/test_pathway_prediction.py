from unittest import TestCase

from plants_sm.pathway_prediction.pathway_prediction import PlantPathwayClassifier


class TestPathwayPrediction(TestCase):

    def test_predict_pathways(self):
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
        predictions = PlantPathwayClassifier('KEGG').predict(smiles[0])
        self.assertEqual(predictions[0], "map00940")

    def test_predict_pathways_ecs(self):
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
        predictions = PlantPathwayClassifier('KEGG').predict_ec_numbers_in_pathway(smiles[0])
        self.assertEqual(len(predictions), 31)


