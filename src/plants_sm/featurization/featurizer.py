from abc import ABCMeta, abstractmethod
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from plants_sm.data_structures.dataset import Dataset


class FeaturesGenerator(metaclass=ABCMeta):

    def __init__(self):
        pass

    def featurize(self, dataset: Dataset):
        parallel_callback = Parallel(n_jobs=self.n_jobs)
        new_x, _ = zip(*parallel_callback(delayed(self._featurize)(instance)
                                          for instance in dataset.instances))

        new_x = pd.concat(new_x, axis=0)
        dataset.dataframe = pd.concat((dataset.dataframe, new_x), axis=1)
        return dataset

    @abstractmethod
    def _featurize(self, instance: Any):
        pass
