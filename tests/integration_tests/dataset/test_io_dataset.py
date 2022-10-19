from unittest import skip

from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_structures.dataset import SingleInputDataset
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset


class TestIODataset(TestDataset):

    def test_read_excel_to_dataset(self):
        dataset = SingleInputDataset.from_excel(self.excel_to_read, sheet_name="DrugInfo",
                                                representation_field="SMILES",
                                                labels_field="TargetedMZ")
        self.assertEqual(list(dataset.get_instances().values())[0], 'Nc1nc(NC2CC2)c2ncn(C3C=CC(CO)C3)c2n1')
        self.assertIn(286.154209, dataset.y)
        self.assertEqual(dataset.representation_fields, {'place_holder': 'SMILES'})
        self.assertEqual(dataset.labels_names, ["TargetedMZ"])
        with self.assertRaises(AttributeError):
            print(dataset.X)

    @skip("Not implemented yet")
    def test_write_excel_from_dataset(self):
        dataset = SingleInputDataset.from_excel(self.excel_to_read, sheet_name="DrugInfo",
                                                representation_field="SMILES", labels_field="TargetedMZ")
        dataset.to_excel(self.df_path_to_write_xlsx, sheet_name="DrugInfo")

        dataset = SingleInputDataset.from_excel(self.df_path_to_write_xlsx, representation_field="SMILES",
                                                sheet_name="DrugInfo",
                                                labels_field="TargetedMZ")
        self.assertEqual(list(dataset.get_instances().values())[0], 'Nc1nc(NC2CC2)c2ncn(C3C=CC(CO)C3)c2n1')
        self.assertIn(286.154209, dataset.y)
        self.assertEqual(dataset.representation_fields, "SMILES")
        self.assertEqual(dataset.labels_names, ["TargetedMZ"])
        print(dataset.identifiers)
        with self.assertRaises(AttributeError):
            print(dataset.features)

    def test_read_csv_to_dataset(self):
        dataset = SingleInputDataset.from_csv(file_path=self.csv_to_read, representation_field="sequence",
                                              instances_ids_field="id")
        self.assertEqual(list(dataset.get_instances().values())[0],
                         'MGWVGKKKSTAGQLAGTANELTKEVLERAVHRESPVIRPDVVVGIPAVDRRPKQ')
        self.assertEqual(dataset.representation_fields, {'place_holder': 'sequence'})
        self.assertEqual(dataset.instances_ids_field, "id")
        with self.assertRaises(AttributeError):
            features = dataset.features

    @skip("Not implemented yet")
    def test_write_csv_from_dataset(self):
        dataset = SingleInputDataset.from_csv(self.csv_to_read, representation_field="sequence",
                                              instances_ids_field="id")

        dataset.to_csv(self.df_path_to_write_csv, index=True)

        dataset2 = SingleInputDataset.from_csv(self.csv_to_read, representation_field="sequence",
                                               instances_ids_field="id")

        self.assertEqual(dataset2.representation_fields, {'place_holder': 'sequence'})
        self.assertEqual(dataset2.instances_ids_field, "id")
        with self.assertRaises(AttributeError):
            print(dataset.features)

    def test_read_csv_to_multi_line_dataset(self):
        self.dataset_400_instances = MultiInputDataset.from_csv(self.multi_input_dataset_csv,
                                                                representation_fields={"proteins": "SEQ",
                                                                                       "ligands": "SUBSTRATES"},
                                                                instances_ids_field={"interaction": "ids"},
                                                                labels_field="LogSpActivity")

        print(self.dataset_400_instances.instances_ids_field)
        print(self.dataset_400_instances.instances["ligands"])
