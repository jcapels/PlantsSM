from abc import ABCMeta, abstractmethod


class UnsupervisedModel(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._fitted = None

    def fit(self, x):
        self._fit(x)
        return self

    def transform(self, x):
        return self._transform(x)

    def predict(self, x):
        return self._predict(x)

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def fit_predict(self, x):
        return self.fit(x).predict(x)

    @property
    def fitted(self):
        return self._fitted

    @fitted.setter
    def fitted(self, value: bool):
        self._fitted = value

    @abstractmethod
    def _fit(self, x):
        pass

    @abstractmethod
    def _transform(self, x):
        pass

    @abstractmethod
    def _predict(self, x):
        pass

