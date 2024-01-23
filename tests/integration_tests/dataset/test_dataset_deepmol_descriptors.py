from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD
from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors


class TestDatasetDeepMol(TestDataset):

    def test_create_dataset_deep_mol(self):
        dataset = SingleInputDataset(dataframe=self.compounds_dataframe, representation_field="sequence")
        deepmol_wrapper = DeepMolDescriptors()
        deepmol_wrapper.fit_transform(dataset)

        self.assertEqual(dataset.dataframe.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 2048)
        self.assertEqual(dataset.features_fields[PLACEHOLDER_FIELD][0], "morgan_0")
        self.assertEqual(dataset.instances_ids_field, "identifier")

    def test_create_dataset_deep_mol_2d_descriptors(self):
        dataset = SingleInputDataset(dataframe=self.compounds_dataframe, representation_field="sequence")
        deepmol_wrapper = DeepMolDescriptors(preset="2d_descriptors")
        deepmol_wrapper.fit_transform(dataset)

        self.assertEqual(dataset.dataframe.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 210)
        self.assertEqual(dataset.features_fields[PLACEHOLDER_FIELD][0], "MaxAbsEStateIndex")
        self.assertEqual(dataset.instances_ids_field, "identifier")
