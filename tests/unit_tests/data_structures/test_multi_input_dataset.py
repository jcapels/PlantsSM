from copy import copy
from unittest import TestCase

import pandas as pd

from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset


class TestMultiInputDataset(TestCase):

    def test_multi_input_dataset(self) -> None:
        dataframe = pd.DataFrame(columns=['ids_interaction', 'ids_a', 'a', 'ids_b', 'b', 'labels'])

        dataframe.loc[0] = ["2", "1", "representation2", "1", "representation5", 1]
        dataframe.loc[1] = ["3", "4", "representation1", "3", "representation4", 0]
        dataframe.loc[2] = ["6", "0", "representation", "6", "representation2", 1]

        dataset = MultiInputDataset(dataframe,
                                    representation_field={"proteins": "a", "compounds": "b"},
                                    instances_ids_field={"proteins": "ids_a", "compounds": "ids_b",
                                                         "interaction": "ids_interaction"},
                                    labels_field='labels')

        self.assertEqual(list(dataset.identifiers), ['2', '3', '6'])
        self.assertEqual(list(dataset.get_instances('proteins').values()), ['representation2', 'representation1',
                                                                            'representation'])

        self.assertEqual(list(dataset.get_instances('compounds').values()), ['representation5', 'representation4',
                                                                             'representation2'])

        self.assertEqual(list(dataset.y), [1, 0, 1])

        new_dataset = copy(dataset)

        new_dataset.select(['2', '3'], "interaction")
        self.assertEqual(list(new_dataset.identifiers), ['2', '3'])

        dataset.select(['1', '0'], "proteins")
        self.assertEqual(list(dataset.identifiers), ['2', '6'])

    def test_with_repetitive_instances(self) -> None:
        dataframe = pd.DataFrame(columns=['ids_interaction', 'ids_a', 'a', 'ids_b', 'b', 'labels'])

        dataframe.loc[0] = ["2", "1", "representation2", "1", "representation5", 1]
        dataframe.loc[1] = ["3", "4", "representation1", "1", "representation5", 0]
        dataframe.loc[2] = ["6", "0", "representation", "6", "representation2", 1]

        dataset = MultiInputDataset(dataframe,
                                    representation_field={"proteins": "a", "compounds": "b"},
                                    instances_ids_field={"proteins": "ids_a", "compounds": "ids_b",
                                                         "interaction": "ids_interaction"},
                                    labels_field='labels')

        self.assertEqual(list(dataset.identifiers), ['2', '3', '6'])
        self.assertEqual(list(dataset.get_instances('proteins').values()), ['representation2', 'representation1',
                                                                            'representation'])

        self.assertEqual(list(dataset.get_instances('compounds').values()), ['representation5',
                                                                             'representation2'])

    def test_merge(self):
        dataframe = pd.DataFrame(columns=['ids_interaction', 'ids_a', 'a', 'ids_b', 'b', 'labels'])

        dataframe.loc[0] = ["2", "1", "representation2", "1", "representation5", 1]
        dataframe.loc[1] = ["3", "4", "representation1", "1", "representation5", 0]
        dataframe.loc[2] = ["6", "0", "representation", "6", "representation2", 1]

        dataset = MultiInputDataset(dataframe,
                                    representation_field={"proteins": "a", "compounds": "b"},
                                    instances_ids_field={"proteins": "ids_a", "compounds": "ids_b",
                                                         "interaction": "ids_interaction"},
                                    labels_field='labels')

        dataframe2 = pd.DataFrame(columns=['ids_interaction', 'ids_a', 'a', 'ids_b', 'b', 'labels'])
        dataframe2.loc[0] = ["2", "1", "representation2", "1", "representation5", 1]
        dataframe2.loc[1] = ["3", "4", "representation1", "1", "representation5", 0]
        dataframe2.loc[2] = ["6", "0", "representation", "6", "representation2", 1]

        dataset2 = MultiInputDataset(dataframe2,
                                     representation_field={"proteins": "a", "compounds": "b"},
                                     instances_ids_field={"proteins": "ids_a", "compounds": "ids_b",
                                                          "interaction": "ids_interaction"},
                                     labels_field='labels')

        dataset.merge(dataset2)

        self.assertEqual(dataset.y.shape[0], 6)
        self.assertEqual(dataset.y[0], 1)
        self.assertEqual(dataset.y[1], 0)
        self.assertEqual(dataset.y[2], 1)
        self.assertEqual(dataset.y[3], 1)
        self.assertEqual(dataset.y[4], 0)
        self.assertEqual(dataset.y[5], 1)
