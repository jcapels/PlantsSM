import os
from unittest import TestCase

from plants_sm.data_structures.dataset import PandasDataset
from tests import TEST_DIR


class TestIODataset(TestCase):

    def setUp(self) -> None:
        self.excel_to_read = os.path.join(TEST_DIR, "data", "drug_list.xlsx")
        self.csv_to_read = os.path.join(TEST_DIR, "data", "proteins.csv")

        self.df_path_to_write_csv = os.path.join(TEST_DIR, "data", "test.csv")
        self.df_path_to_write_xlsx = os.path.join(TEST_DIR, "data", "test.xlsx")

    def tearDown(self) -> None:
        paths_to_remove = [self.df_path_to_write_csv,
                           self.df_path_to_write_xlsx]

        for path in paths_to_remove:
            if os.path.exists(path):
                os.remove(path)

    def test_read_excel_to_dataset(self):
        dataset = PandasDataset(representation_field="SMILES",
                                labels_field="TargetedMZ")
        dataset.from_excel(self.excel_to_read, sheet_name="DrugInfo")
        self.assertEqual(dataset.instances[0], 'Nc1nc(NC2CC2)c2ncn(C3C=CC(CO)C3)c2n1')
        self.assertIn(286.154209, dataset.labels)
        self.assertEqual(dataset.representation_field, "SMILES")
        self.assertEqual(dataset.labels_names, ["TargetedMZ"])
        with self.assertRaises(ValueError):
            features = dataset.features

    def test_write_excel_from_dataset(self):
        dataset = PandasDataset(representation_field="sequence",
                                labels_field="id")
        dataset.from_excel(self.excel_to_read, sheet_name="DrugInfo")
        dataset.to_excel(self.df_path_to_write_xlsx, sheet_name="DrugInfo")

        dataset = PandasDataset(representation_field="SMILES",
                                labels_field="TargetedMZ")
        dataset.from_excel(self.df_path_to_write_xlsx, sheet_name="DrugInfo")
        self.assertEqual(dataset.instances[0], 'Nc1nc(NC2CC2)c2ncn(C3C=CC(CO)C3)c2n1')
        self.assertIn(286.154209, dataset.labels)
        self.assertEqual(dataset.representation_field, "SMILES")
        self.assertEqual(dataset.labels_names, ["TargetedMZ"])
        with self.assertRaises(ValueError):
            features = dataset.features

    def test_read_csv_to_dataset(self):
        dataset = PandasDataset(representation_field="sequence",
                                instances_ids_field="id")
        dataset.from_csv(self.csv_to_read)
        self.assertEqual(dataset.instances[0], 'MGWVGKKKSTAGQLAGTANELTKEVLERAVHRESPVIRPDVVVGIPAVDRRPKQ')
        self.assertEqual(dataset.representation_field, "sequence")
        self.assertEqual(dataset.instances_ids_field, "id")
        with self.assertRaises(ValueError):
            features = dataset.features

    def test_write_csv_from_dataset(self):
        dataset = PandasDataset(representation_field="sequence",
                                instances_ids_field="id")
        dataset.from_csv(self.csv_to_read)

        dataset.to_csv(self.df_path_to_write_csv, index=False)

        dataset = PandasDataset(representation_field="sequence",
                                instances_ids_field="id")
        dataset.from_csv(self.df_path_to_write_csv)

        self.assertEqual(dataset.instances[0], 'MGWVGKKKSTAGQLAGTANELTKEVLERAVHRESPVIRPDVVVGIPAVDRRPKQ')
        self.assertEqual(dataset.representation_field, "sequence")
        self.assertEqual(dataset.instances_ids_field, "id")
        with self.assertRaises(ValueError):
            features = dataset.features
