from collections import ChainMap
from typing import Callable

from joblib import Parallel, delayed

from plants_sm.data_structures.dataset import Dataset


def transform_instances(n_jobs: int, dataset: Dataset, func: Callable) -> Dataset:
    parallel_callback = Parallel(n_jobs=n_jobs)
    len_instances = len(dataset.instances)
    res = parallel_callback(
        delayed(func)(dataset.instances[i], dataset.identifiers[i])
        for i in range(len_instances))

    sequences_dict = dict(ChainMap(*res))
    dataset.dataframe.loc[:, dataset.representation_field] = dataset.dataframe.index.map(
        sequences_dict)

    return dataset