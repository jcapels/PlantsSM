import pandas as pd
from pandas import DataFrame

from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.featurization.propythia.propythia_descriptors.presets import DESCRIPTORS_PRESETS


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
        if self.descriptor not in DESCRIPTORS_PRESETS:
            raise ValueError(f'Preset {self.descriptor} is not available.')
        self.descriptors = [descriptor() for descriptor in DESCRIPTORS_PRESETS[self.descriptor]]
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
        features_names = []
        features_list = []
        for descriptor in self.descriptors:
            features = descriptor(protein_sequence, **self.kwargs)
            features_names.extend(descriptor.get_features_out())
            features_list.extend(features)

        features_df = DataFrame([features_list], index=[0], columns=features_names)
        if self.features_names is None:
            self.features_names = features_names
        features_df[identifier_field_name] = [identifier]
        return features_df
