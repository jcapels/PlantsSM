import pandas as pd
from pandas import DataFrame

from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.featurization.propythia_functions.protein_descriptors import PropythiaDescriptors


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
        self.general_descriptor = PropythiaDescriptors()
        super().__init__(**kwargs)

    def _featurize(self, protein_sequence: str, identifier: str, identifier_field_name: str) -> pd.DataFrame:
        """
        The method _featurize will generate the desired features for a given protein sequence

        Parameters
        ----------
        protein_sequence: str
            protein sequence string

        identifier: str
            protein identifier

        Returns
        -------
        dataframe with features: pd.DataFrame
        """
        func = getattr(self.general_descriptor, self.descriptor)
        features = func(protein_sequence, **self.kwargs)
        features_df = DataFrame(features, index=[0])
        if self.features_names is None:
            self.features_names = list(features_df.columns)
        features_df[identifier_field_name] = [protein_sequence]
        return features_df
