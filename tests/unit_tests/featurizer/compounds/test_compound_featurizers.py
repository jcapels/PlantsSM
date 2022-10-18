from unittest import TestCase
from unittest.mock import MagicMock

from pandas import DataFrame


class TestCompoundFeaturizers(TestCase):

    def setUp(self) -> None:
        self.dataset = MagicMock()
        self.dataset.dataframe = DataFrame(columns=["SMILES", "identifiers"])
        self.dataset.features_dataframe = DataFrame()
        self.dataset.representation_field = "sequence"
        self.dataset.identifiers = ["0", "1"]
        self.dataset.instances_ids_field = "identifiers"
        self.dataset.features_fields = None
        self.dataset.instances = [
            "CCCCCCCO", "C1=CC=C(C=C1)C=O"
        ]
        self.dataset.dataframe["SMILES"] = self.dataset.instances
        self.dataset.dataframe["identifiers"] = self.dataset.identifiers
