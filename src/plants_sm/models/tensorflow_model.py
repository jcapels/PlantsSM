import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.models._utils import _convert_proba_to_unified_form
from plants_sm.models.constants import REGRESSION, QUANTILE
from plants_sm.models.model import Model

import tensorflow as tf


class TensorflowModel(Model):

    def __init__(self, model: tf.keras.Model, epochs: int, batch_size: int, problem_type: str,
                 callbacks: list = None):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.problem_type = problem_type

    def _preprocess_data(self, dataset: Dataset, **kwargs):
        pass

    def _fit_data(self, dataset: Dataset, validation_dataset: Dataset):
        self._history = self.model.fit(dataset.X.values(), dataset.y,
                                       epochs=self.epochs,
                                       batch_size=self.batch_size,
                                       callbacks=self.callbacks,
                                       validation_data=(validation_dataset.X.values(), validation_dataset.y))

    def _predict_proba(self, dataset: Dataset):
        self.model.predict_proba(dataset.X.values())

    def _predict(self, dataset: Dataset):
        if self.problem_type in [REGRESSION, QUANTILE]:
            y_pred = self.model.predict(dataset.X.values())
            return y_pred
        else:
            predictions = self.model.predict(dataset.X.values())
            return _convert_proba_to_unified_form(self.problem_type, np.array(predictions))

    def _save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model.load(path)

    @property
    def history(self):
        return self._history
