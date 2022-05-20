import pandas as pd
from pandas import DataFrame

from plants_sm.featurization.featurizer import FeaturesGenerator
from propythia.descriptors import Descriptor


class PropythiaWrapper(FeaturesGenerator):

    def __init__(self, descriptor: str, **kwargs):
        """
        Constructor

        Parameters
        ----------
        descriptor: str
        kwargs: additional arguments

        """
        self.descriptor = descriptor
        self.general_descriptor = Descriptor("")
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def _featurize(self, protein_sequence: str) -> pd.DataFrame:
        """
        The method _featurize will generate the desired features for a given protein sequence

        Parameters
        ----------
        protein_sequence: str
            protein sequence string

        Returns
        -------
        dataframe with features: pd.DataFrame
        """
        self.general_descriptor.ProteinSequence = protein_sequence
        func = getattr(self.general_descriptor, self.descriptor)
        features = func(**self.kwargs)
        features_df = DataFrame(features, index=[0])
        return features_df
