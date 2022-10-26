from collections import ChainMap
from typing import Callable

from joblib import Parallel, delayed

from plants_sm.data_structures.dataset import Dataset

import contextlib
import joblib
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def transform_instances(n_jobs: int, dataset: Dataset, func: Callable, instance_type: str) -> Dataset:
    parallel_callback = Parallel(n_jobs=n_jobs, prefer="threads")
    instances = dataset.get_instances(instance_type)
    with tqdm_joblib(tqdm(desc="My calculation", total=len(instances.items()))):
        res = parallel_callback(
            delayed(func)(instance_representation, instance_id)
            for instance_id, instance_representation in instances.items())

    sequences_dict = dict(res)
    dataset.instances[instance_type] = sequences_dict

    return dataset
