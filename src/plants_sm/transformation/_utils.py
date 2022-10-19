from collections import ChainMap
from typing import Callable

from joblib import Parallel, delayed

from plants_sm.data_structures.dataset import Dataset


def transform_instances(n_jobs: int, dataset: Dataset, func: Callable, instance_type: str) -> Dataset:
    parallel_callback = Parallel(n_jobs=n_jobs)
    instances = dataset.get_instances(instance_type)
    res = parallel_callback(
        delayed(func)(instance_representation, instance_id)
        for instance_id, instance_representation in instances.items())

    sequences_dict = dict(ChainMap(*res))
    dataset.instances[instance_type] = sequences_dict

    return dataset
