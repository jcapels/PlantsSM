from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_structures.dataset import PandasDataset
from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper


class TestDatasetPropythia(TestDataset):

    def test_create_dataset_propythia(self):
        dataset = PandasDataset(dataframe=self.dataframe, representation_field="sequence")
        propythia = PropythiaWrapper(preset="all")
        propythia.fit_transform(dataset)

        self.assertEqual(dataset.dataframe.shape[0], 2)
        self.assertEqual(dataset.features.shape[1], 9596)
        self.assertEqual(dataset.features_fields[0], "length")
        self.assertEqual(dataset.instances_ids_field, "identifier")

    def test_different_orders_dictionary_dataframe(self):
        dataset = PandasDataset(representation_field="sequence",
                                instances_ids_field="id")
        dataset.from_csv(self.csv_to_read)

        propythia = PropythiaWrapper(preset="physicochemical")
        propythia.fit_transform(dataset)

        self.assertEqual(list(dataset.dataframe.index), list(dataset.features_dataframe.index))
