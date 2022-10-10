from functools import wraps


def call_set_features_names(func):
    """
    Decorator that calls the method set_features_names before the method transform

    Parameters
    ----------
    func: _transform function always
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            if not self.features_names:
                self.set_features_names()
            res = func(self, *args, **kwargs)
            return res

        except ValueError as e:
            raise e

    return wrapper
