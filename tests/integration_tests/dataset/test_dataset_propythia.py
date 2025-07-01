from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.ml.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD
from plants_sm.ml.featurization.proteins.propythia.propythia import PropythiaWrapper


class TestDatasetPropythia(TestDataset):

    def test_create_dataset_propythia(self):
        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence")
        propythia = PropythiaWrapper(preset="all")
        propythia.fit_transform(dataset)

        self.assertEqual(dataset.dataframe.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 9595)
        self.assertEqual(dataset.features_fields[PLACEHOLDER_FIELD][0], "charge")
        self.assertEqual(dataset.instances_ids_field, "identifier")

    def test_different_orders_dictionary_dataframe(self):
        dataset = SingleInputDataset.from_csv(self.csv_to_read, representation_field="sequence",
                                              instances_ids_field="id")

        propythia = PropythiaWrapper(preset="physicochemical")
        propythia.fit_transform(dataset)

        self.assertAlmostEqual(float(dataset.X[0, 0]), 4.031, delta=0.1)
