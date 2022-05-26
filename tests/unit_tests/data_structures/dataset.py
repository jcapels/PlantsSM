from unittest import TestCase

import pandas as pd

from plants_sm.data_structures.dataset import PandasDataset


class TestDataset(TestCase):

    def test_pandas_dataset(self):
        """
        Test the PandasDataset class.
        """
        dataframe = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe.loc[0] = ["representation2", 2, 3]
        dataframe.loc[1] = ["representation1", 5, 6]
        dataframe.loc[2] = ["representation", 8, 9]

        dataset = PandasDataset(dataframe, representation_field='a', labels_field='c', features_field='b')
        self.assertTrue(dataset.representation_field == 'a')
        self.assertTrue(dataset.features_names == ['b'])
        self.assertTrue(dataset.labels_names == ['c'])
        self.assertTrue(all(representation in dataset.instances for representation in ['representation2',
                                                                                       'representation1',
                                                                                       'representation']))
        self.assertTrue(all(feature in dataset.features for feature in [2, 5, 8]))
        self.assertTrue(all(label in dataset.labels for label in [3, 6, 9]))

        dataset = PandasDataset(dataframe, representation_field='a', labels_field='c', features_field=[1])
        self.assertTrue(dataset.features_names == ['b'])

