from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from pandas import DataFrame

from plants_sm.ml.data_structures.dataset import PLACEHOLDER_FIELD, SingleInputDataset


class TestCompoundFeaturizers(TestCase):
    def X(self):
        return np.array(list(self.dataset.features[PLACEHOLDER_FIELD].values()), dtype=np.int32)

    def add_features(self, instance_type, features):
        self.dataset.features[instance_type] = features

    def setUp(self) -> None:
        self.dataset = MagicMock(spec=SingleInputDataset)
        self.dataset.dataframe = DataFrame(columns=["SMILES", "identifiers"])
        self.dataset.features_dataframe = DataFrame()
        self.dataset.representation_field = "sequence"
        self.dataset.identifiers = ["0", "1"]
        self.dataset.instances_ids_field = "identifiers"
        self.dataset.features_fields = {}
        self.dataset.features = {}

        self.dataset.instances = {PLACEHOLDER_FIELD:
            {
                "0": "CCCCCCCO", "1": "C1=CC=C(C=C1)C=O"
            }}

        self.dataset.get_instances.return_value = {
            "0": "CCCCCCCO", "1": "C1=CC=C(C=C1)C=O"
        }
        self.dataset.dataframe["SMILES"] = self.dataset.instances[PLACEHOLDER_FIELD].values()
        self.dataset.dataframe["identifiers"] = self.dataset.identifiers
        self.dataset.X = self.X
        self.dataset.add_features = self.add_features
        self.dataset.batch_size = None
