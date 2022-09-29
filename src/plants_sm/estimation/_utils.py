from functools import wraps


def fit_status(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # TODO: implement try/except block to catch for fitting errors rather than ValueError
        try:
            # it is required to set fitted twice.
            # the first time allows to set the estimated parameters,
            # as these can be accessed without fitting the estimator
            # the second time is to set the status to True,
            # even if some parameters are internally altered during the fitting process due to parameter processing
            self._fitted = True
            res = func(self, *args, **kwargs)
            self._fitted = True
            return res

        except ValueError as e:
            self._fitted = False
            raise e

    return wrapper