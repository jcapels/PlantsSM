from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch
from pandas import DataFrame

from plants_sm.ml.data_structures.dataset.single_input_dataset import PLACEHOLDER_FIELD, SingleInputDataset


class TestReactionFeaturizers(TestCase):

    def X(self):
        return np.array(list(self.dataset.features[PLACEHOLDER_FIELD].values()))

    def add_features(self, instance_type, features):
        self.dataset.features[instance_type] = features

    def setUp(self) -> None:
        self.dataset = MagicMock(spec=SingleInputDataset)
        self.dataset.dataframe = DataFrame(columns=["sequence", "identifiers"])
        self.dataset.features_dataframe = DataFrame()
        self.dataset.representation_field = "sequence"
        self.dataset.identifiers = ["0", "1"]
        self.dataset.instances_ids_field = "identifiers"
        self.dataset.features_fields = {}
        self.dataset.features = {}
        self.dataset.get_instances.return_value = {"0":
                                                       "O=O.[H]O[H].[H][C@]1(C2=CN=C(O)C=C2)CCC[NH+]1C>>C[NH2+]CCCC(=O)C1=CN=C(O)C=C1.[H]OO[H]",
                                                   "1":
                                                      "COC1=CC(/C=C/C(=O)OCC[N+](C)(C)C)=CC(OC)=C1O.[H]O[H]>>COC1=CC(/C=C/C(=O)[O-])=CC(OC)=C1O.C[N+](C)(C)CCO.[H+]" 
                                                   }
        self.dataset.dataframe["identifiers"] = self.dataset.identifiers
        torch.seed()
        self.dataset.X = self.X
        self.dataset.add_features = self.add_features
        self.dataset.batch_size = None
