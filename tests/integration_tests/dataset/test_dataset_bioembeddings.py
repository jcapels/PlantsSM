from copy import copy

import pandas as pd

from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD
from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert


class TestDatasetBioembeddings(TestDataset):


    def test_create_dataset_prot_bert(self):
        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence")
        unirep = ProtBert()
        unirep.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 1024)
        self.assertEqual(dataset.features_fields[PLACEHOLDER_FIELD][0], "prot_bert_1")
        self.assertEqual(dataset.instances_ids_field, "identifier")

        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence",
                                     instances_ids_field="ids")

        unirep = ProtBert(output_shape_dimension=3)
        unirep.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 453)
        self.assertEqual(dataset.X.shape[2], 1024)
        self.assertEqual(dataset.features_fields[PLACEHOLDER_FIELD][0], "prot_bert_1")
        self.assertEqual(dataset.instances_ids_field, "ids")
