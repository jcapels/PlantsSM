import os
from unittest import TestCase, skip

import pandas as pd

from plants_sm.pathway_prediction.esi_annotator import ESM1bESIAnnotator, ESM2ESIAnnotator, ProtBertESIAnnotator
from tests import TEST_DIR

@skip("Requires model download")
class TestESIAnnotator(TestCase):

    def setUp(self):
        
        self.data_path = os.path.join(TEST_DIR,
                                      "data", 
                                      "multi_input_dataset.csv")
        self.annotator = ProtBertESIAnnotator()
        self.non_valid_smiles_data_path = os.path.join(TEST_DIR,
                                      "data", 
                                      "non_valid_smiles_multi_input_dataset.csv")
        
    def test_esi_annotator(self):
        data = pd.read_csv(self.data_path)
        unique_compounds = pd.unique(data.iloc[:, 2]) 
        solution = self.annotator.annotate_from_file(self.data_path, "csv")
        self.assertEqual(len(solution.substrate_protein_solutions), len(unique_compounds))
        self.assertEqual(len(solution.get_score(39)), 2)
        self.assertGreater(solution.get_score(39)[0][1], solution.get_score(39)[1][1])
        self.assertIsInstance(solution.dataframe_with_solutions, pd.DataFrame)
        self.assertEqual(solution.dataframe_with_solutions.shape[0], data.shape[0])

    def test_non_valid_compounds(self):
        
        data = pd.read_csv(self.non_valid_smiles_data_path)
        unique_compounds = pd.unique(data.iloc[:, 2]) 
        solution = self.annotator.annotate_from_file(self.non_valid_smiles_data_path, "csv")
        self.assertEqual(len(solution.substrate_protein_solutions), len(unique_compounds)-1)
        self.assertEqual(len(solution.get_score(39)), 2)
        self.assertGreater(solution.get_score(39)[0][1], solution.get_score(39)[1][1])

@skip("Requires model download")
class TestESM1bESIAnnotator(TestCase):

    def setUp(self):
        
        self.data_path = os.path.join(TEST_DIR,
                                      "data", 
                                      "multi_input_dataset.csv")
        self.annotator = ESM1bESIAnnotator()
        self.non_valid_smiles_data_path = os.path.join(TEST_DIR,
                                      "data", 
                                      "non_valid_smiles_multi_input_dataset.csv")
        
    def test_esi_annotator(self):
        data = pd.read_csv(self.data_path)
        unique_compounds = pd.unique(data.iloc[:, 2]) 
        solution = self.annotator.annotate_from_file(self.data_path, "csv")
        self.assertEqual(len(solution.substrate_protein_solutions), len(unique_compounds))
        self.assertEqual(len(solution.get_score(39)), 2)
        self.assertGreater(solution.get_score(39)[0][1], solution.get_score(39)[1][1])
        self.assertIsInstance(solution.dataframe_with_solutions, pd.DataFrame)
        self.assertEqual(solution.dataframe_with_solutions.shape[0], data.shape[0])

    def test_non_valid_compounds(self):
        
        data = pd.read_csv(self.non_valid_smiles_data_path)
        unique_compounds = pd.unique(data.iloc[:, 2]) 
        solution = self.annotator.annotate_from_file(self.non_valid_smiles_data_path, "csv")
        self.assertEqual(len(solution.substrate_protein_solutions), len(unique_compounds)-1)
        self.assertEqual(len(solution.get_score(39)), 2)
        self.assertGreater(solution.get_score(39)[0][1], solution.get_score(39)[1][1])

@skip("Requires model download")
class TestESM2ESIAnnotator(TestCase):

    def setUp(self):
        
        self.data_path = os.path.join(TEST_DIR,
                                      "data", 
                                      "multi_input_dataset.csv")
        self.annotator = ESM2ESIAnnotator()
        self.non_valid_smiles_data_path = os.path.join(TEST_DIR,
                                      "data", 
                                      "non_valid_smiles_multi_input_dataset.csv")
        
    def test_esi_annotator(self):
        data = pd.read_csv(self.data_path)
        unique_compounds = pd.unique(data.iloc[:, 2]) 
        solution = self.annotator.annotate_from_file(self.data_path, "csv")
        self.assertEqual(len(solution.substrate_protein_solutions), len(unique_compounds))
        self.assertEqual(len(solution.get_score(39)), 2)
        self.assertGreater(solution.get_score(39)[0][1], solution.get_score(39)[1][1])
        self.assertIsInstance(solution.dataframe_with_solutions, pd.DataFrame)
        self.assertEqual(solution.dataframe_with_solutions.shape[0], data.shape[0])

    def test_non_valid_compounds(self):
        
        data = pd.read_csv(self.non_valid_smiles_data_path)
        unique_compounds = pd.unique(data.iloc[:, 2]) 
        solution = self.annotator.annotate_from_file(self.non_valid_smiles_data_path, "csv")
        self.assertEqual(len(solution.substrate_protein_solutions), len(unique_compounds)-1)
        self.assertEqual(len(solution.get_score(39)), 2)
        self.assertGreater(solution.get_score(39)[0][1], solution.get_score(39)[1][1])