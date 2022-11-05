from unittest import skip

from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD
from plants_sm.featurization.compounds.map4_fingerprint import MAP4Fingerprint
from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper


# @skip("Requires conda")
class TestDatasetMAP4(TestDataset):

    def test_create_dataset_deep_mol(self):
        dataset = SingleInputDataset(dataframe=self.compounds_dataframe, representation_field="sequence")
        deepmol_wrapper = MAP4Fingerprint()
        deepmol_wrapper.fit_transform(dataset)

        self.assertEqual(dataset.dataframe.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 1024)
        self.assertEqual(dataset.features_fields[PLACEHOLDER_FIELD][0], "map4_fingerprints_1")
        self.assertEqual(dataset.instances_ids_field, "identifier")

    def test_multi_input_dataset_map4_fingerprint(self):
        deepmol_wrapper = MAP4Fingerprint()
        deepmol_wrapper.fit_transform(self.multi_input_dataset, "ligands")

        propythia = PropythiaWrapper(preset="all")
        propythia.fit_transform(self.multi_input_dataset, "proteins")

        self.assertEqual(self.multi_input_dataset.X["ligands"].shape[0], 8)
        self.assertEqual(self.multi_input_dataset.X["ligands"].shape[1], 1024)