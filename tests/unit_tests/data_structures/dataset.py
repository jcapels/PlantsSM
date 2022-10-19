from unittest import TestCase

import numpy as np
import pandas as pd

from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset


class TestDataset(TestCase):

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
        self.assertTrue(dataset._features_fields == ['b'])
        self.assertTrue(dataset.labels_names == ['c'])
        self.assertTrue(all(representation in dataset.instances for representation in ['representation2',
                                                                                       'representation1',
                                                                                       'representation']))
        self.assertTrue(all(feature in dataset.X for feature in [2, 5, 8]))
        self.assertTrue(all(label in dataset.y for label in [3, 6, 9]))

        dataset = SingleInputDataset(dataframe, representation_field='a', labels_field='c', features_fields=[1])
        self.assertTrue(dataset._features_fields == ["b"])
        self.assertEqual(dataset.X.shape[1], 1)

    def test_slice_as_features_field(self):
        dataframe = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe.loc[0] = ["representation2", 2, 3]
        dataframe.loc[1] = ["representation1", 5, 6]
        dataframe.loc[2] = ["representation", 8, 9]

        dataset = SingleInputDataset(dataframe, representation_field='a', features_fields=slice(1, None))
        self.assertTrue(dataset.X.shape[1] == 2)

