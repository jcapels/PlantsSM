from unittest import TestCase
from unittest.mock import MagicMock

from pandas import DataFrame

from plants_sm.data_standardization.compounds.deepmol_standardizers import DeepMolStandardizer
from plants_sm.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD


class TestStandardizer(TestCase):

    def setUp(self) -> None:
        self.dataset = MagicMock(spec=SingleInputDataset)
        self.dataset.dataframe = DataFrame(columns=["sequence", "identifiers"])
        self.dataset.features_dataframe = DataFrame()
        self.dataset.representation_field = "sequence"
        self.dataset.identifiers = ["0", "1"]
        self.dataset.instances_ids_field = "identifiers"
        self.dataset.features_fields = None
        self.dataset.instances = {PLACEHOLDER_FIELD: {"0":
                                                          "CCCCCCCO",
                                                      "1":
                                                          "C1=CC=C(C=C1)C=O"
                                                      }}
        self.dataset.batch_size = None
        self.dataset.dataframe["sequence"] = self.dataset.instances

    def test_standardizer(self):
        DeepMolStandardizer().fit_transform(self.dataset)
