import os
from unittest import TestCase, skip

from plants_sm.data_structures.dataset import PandasDataset
from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper
from tests import TEST_DIR


#@skip("Just for performance testing")
class TestReadData(TestCase):

    def setUp(self) -> None:
        csv_to_read = os.path.join(TEST_DIR, "performance_datasets", "aminotransferase_binary.csv")
        self.dataset_400_instances = PandasDataset(representation_field="SEQ", instances_ids_field="ids",
                                                   labels_field="LogSpActivity"). \
            from_csv(csv_to_read)

    def test_featurize_400_instances_propythia(self):
        propythia = PropythiaWrapper(preset="all", n_jobs=8)
        propythia.fit_transform(self.dataset_400_instances)

    def test_featurize_400_instances_propythia_physicochemical(self):
        propythia = PropythiaWrapper(preset="physicochemical", n_jobs=8)
        propythia.fit_transform(self.dataset_400_instances)

    def test_featurize_400_instances_propythia_performance(self):

        propythia = PropythiaWrapper(preset="performance", n_jobs=8)
        propythia.fit_transform(self.dataset_400_instances)