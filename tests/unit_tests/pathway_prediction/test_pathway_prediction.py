import os
from unittest import TestCase, skip
import webbrowser

from plants_sm.pathway_prediction.pathway_prediction import PlantPathwayClassifier
from plants_sm.pathway_prediction.solution import ECSolution

from tests import TEST_DIR

@skip("Need model download")
class TestPathwayPrediction(TestCase):

    def setUp(self):
        self.tomato_genome_annotation = os.path.join(TEST_DIR, "data", "tomato_ec_annotation.csv")
        self.tomato_genome = os.path.join(TEST_DIR, "data", "tomato_proteins.fasta")

    def test_predict_pathways(self):
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
        predictions = PlantPathwayClassifier('KEGG').predict(smiles[0])
        self.assertEqual(predictions[0], "map00940")

    def test_predict_pathways_ecs(self):
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
        predictions = PlantPathwayClassifier('KEGG').predict_ec_numbers_in_pathway(smiles[0])
        self.assertEqual(len(predictions), 31)

    def test_predict_ecs_present_in_pathways(self):
        ec_solution = ECSolution.from_csv_and_fasta(self.tomato_genome_annotation, self.tomato_genome, "accession")
        ec_solution.create_ec_to_entities()
        smiles = ["CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(\C)C2C=C3C(C)(C)CCCC3(C)O2)C(C)(C)CCC1"]
        PlantPathwayClassifier('KEGG').predict_ecs_present_in_pathways(smiles[0], ec_solution)

    def test_predict_pathways_plantcyc(self):
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
        predictions = PlantPathwayClassifier('PlantCyc').predict(smiles[0])
        self.assertEqual(predictions.tolist(), ['PWY-5958','PWY18C3-25'])

    def test_predict_ecs_pathways_plantcyc(self):
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
        predictions, pathways = PlantPathwayClassifier('PlantCyc').predict_ec_numbers_in_pathway(smiles[0])
        self.assertEqual(len(predictions), 6)

    def test_predict_reactions_pathways_plantcyc(self):
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
        ec_solution = ECSolution.from_csv_and_fasta(self.tomato_genome_annotation, self.tomato_genome, "accession")
        predictions = PlantPathwayClassifier('PlantCyc').predict_reactions_present_in_pathways_and_ec_solution(smiles[0], ec_solution)
        self.assertEqual(len(predictions), 9)

    def test_predict_reactions_pathways_kegg(self):
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
        ec_solution = ECSolution.from_csv_and_fasta(self.tomato_genome_annotation, self.tomato_genome, "accession")
        predictions = PlantPathwayClassifier('KEGG').predict_reactions_present_in_pathways_and_ec_solution(smiles[0], ec_solution)
        self.assertEqual(len(predictions), 50)
        

        


