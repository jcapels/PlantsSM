import os
from unittest import TestCase

import pandas as pd

from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset, PLACEHOLDER_FIELD

from tests import TEST_DIR


class TestDataset(TestCase):

    def setUp(self) -> None:
        self.dataset_csv = os.path.join(TEST_DIR, "data", "proteins.csv")

    def test_pandas_dataset(self):
        """
        Test the PandasDataset class.
        """
        dataframe = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe.loc[0] = ["representation2", 2, 3]
        dataframe.loc[1] = ["representation1", 5, 6]
        dataframe.loc[2] = ["representation", 8, 9]

        dataset = SingleInputDataset(dataframe, representation_field='a', labels_field='c', features_fields='b')
        self.assertTrue(dataset.representation_field == 'a')
        self.assertTrue(dataset._features_fields[PLACEHOLDER_FIELD] == ['b'])
        self.assertTrue(dataset._labels_names == ['c'])
        representation_fields = dataset.instances[PLACEHOLDER_FIELD].values()
        self.assertTrue(all(representation in representation_fields for representation in
                            ['representation2',
                             'representation1',
                             'representation']))
        self.assertTrue(all(feature in dataset.X for feature in [2, 5, 8]))
        self.assertTrue(all(label in dataset.y for label in [3, 6, 9]))

        dataset = SingleInputDataset(dataframe, representation_field='a', labels_field='c', features_fields=[1])
        self.assertTrue(dataset._features_fields[PLACEHOLDER_FIELD] == ["b"])
        self.assertEqual(dataset.X.shape[1], 1)

        dataset.select([0, 1])
        self.assertTrue(dataset.X.shape[0] == 2)
        self.assertTrue(dataset.y.shape[0] == 2)
        self.assertEqual(list(dataset.identifiers), [0, 1])

    def test_slice_as_features_field(self):
        dataframe = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe.loc[0] = ["representation2", 2, 3]
        dataframe.loc[1] = ["representation1", 5, 6]
        dataframe.loc[2] = ["representation", 8, 9]

        dataset = SingleInputDataset(dataframe, representation_field='a', features_fields=slice(1, None))
        self.assertTrue(dataset.X.shape[1] == 2)

    def test_read_and_write_to_csv(self):
        """
        Test the read and write to csv methods.
        """
        dataset = SingleInputDataset.from_csv(self.dataset_csv, representation_field="sequence",
                                              instances_ids_field="id",
                                              labels_field="y")

        dataset.to_csv("test.csv", index=False)

        written_dataset = pd.read_csv("test.csv")

        actual_dataset = pd.read_csv(self.dataset_csv)

        self.assertTrue(written_dataset.sequence.equals(actual_dataset.sequence))

        # remove the file
        os.remove("test.csv")

    def test_merge(self):
        """
        Test the merge method.
        """
        dataframe = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe.loc[0] = ["representation2", 2, 3]
        dataframe.loc[1] = ["representation1", 5, 6]
        dataframe.loc[2] = ["representation", 8, 9]

        dataset = SingleInputDataset(dataframe, representation_field='a', labels_field='c', features_fields='b')

        dataframe2 = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe2.loc[0] = ["representation2", 754, 3]
        dataframe2.loc[1] = ["representation1", 53224, 6]
        dataframe2.loc[2] = ["representation", 65437, 9]

        dataset2 = SingleInputDataset(dataframe2, representation_field='a', labels_field='c', features_fields='b')

        dataset.merge(dataset2)

        self.assertTrue(dataset.X.shape == (6, 1))
        self.assertTrue(dataset.y.shape == (6, 1))
        self.assertTrue(dataset.X[0] == 2)
        self.assertTrue(dataset.X[1] == 5)
        self.assertTrue(dataset.X[2] == 8)
        self.assertTrue(dataset.X[3] == 754)
        self.assertTrue(dataset.X[4] == 53224)
        self.assertTrue(dataset.X[5] == 65437)
