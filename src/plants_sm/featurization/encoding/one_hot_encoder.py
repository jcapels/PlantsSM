
import numpy as np

from plants_sm.featurization.encoding.encoder import Encoder


class OneHotEncoder(Encoder):
    """
    Generic class to encode sequences with one-hot encoding.

    Parameters
    ----------
    """
    name = "one_hot_encoder"
    output_shape_dimension = 3

    def _set_tokens(self, i: int, token: str):
        one_hot = np.zeros(len(self.alphabet))
        one_hot[i] = 1
        self.tokens[token] = one_hot

