import os
from unittest import skip

from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD
from plants_sm.featurization.compounds.map4_fingerprint import MAP4Fingerprint
from plants_sm.io import read_csv

from tests import TEST_DIR


@skip("Requires conda")
class TestDatasetMAP4(TestDataset):

    def test_create_dataset_deep_mol(self):
        dataset = SingleInputDataset(dataframe=self.compounds_dataframe, representation_field="sequence")
        deepmol_wrapper = MAP4Fingerprint()
        deepmol_wrapper.fit_transform(dataset)

        self.assertEqual(dataset.dataframe.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 1024)
        self.assertEqual(dataset.features_fields[PLACEHOLDER_FIELD][0], "mp4_fingerprints_1")
        self.assertEqual(dataset.instances_ids_field, "identifier")
