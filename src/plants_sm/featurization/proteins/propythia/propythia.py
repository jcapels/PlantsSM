import pandas as pd
from pandas import DataFrame

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.featurization.proteins.propythia.propythia_descriptors.presets import DESCRIPTORS_PRESETS


class PropythiaWrapper(FeaturesGenerator):
    preset: str

    def _fit(self, dataset: Dataset):
        """
        Abstract method that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        """
        if self.preset not in DESCRIPTORS_PRESETS:
            raise ValueError(f'Preset {self.preset} is not available.')
        self.descriptors = [descriptor() for descriptor in DESCRIPTORS_PRESETS[self.preset]]

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
        features_names = []
        features_list = []
        for descriptor in self.descriptors:
            features = descriptor(protein_sequence)
            features_names.extend(descriptor.get_features_out())
            features_list.extend(features)

        features_df = DataFrame([features_list], index=[0], columns=features_names)
        return features_df