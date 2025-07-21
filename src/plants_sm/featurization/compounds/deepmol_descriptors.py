from typing import Dict

import numpy as np
from rdkit.Chem import MolFromSmiles

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.compounds._presets import DEEPMOL_PRESETS
from plants_sm.featurization.featurizer import FeaturesGenerator


class DeepMolDescriptors(FeaturesGenerator):

    preset: str = "morgan_fingerprints"
    kwargs: Dict = {}

    def set_features_names(self):
        """
        Method to set the names of the features
        """
        self.features_names = self.descriptor.feature_names

    def _fit(self, dataset: Dataset, instance_type: str) -> 'DeepMolDescriptors':
        """
        Method to fit the transformer

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.

        Returns
        -------
        self: DeepMolDescriptors

        """

        if self.preset not in DEEPMOL_PRESETS:
            raise ValueError(f'Preset {self.preset} is not available.')

        descriptor = DEEPMOL_PRESETS[self.preset]
        self.descriptor = descriptor(**self.kwargs)
        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'FeaturesGenerator':
        """
        Method to fit the transformer

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        instance_type: str
            type of the instance to be processed

        Returns
        -------
        self: DeepMolDescriptors

        """

        return self._fit(dataset, instance_type)

    def _featurize(self, molecule: str) -> np.ndarray:
        """
        Method to featurize a molecule

        Parameters
        ----------
        molecule: str
            SMILES string of the molecule to be featurized

        Returns
        -------
        features: np.ndarray
        """

        mol = MolFromSmiles(molecule)

        features = self.descriptor._featurize(mol)

        if not self.features_names:
            if features.shape[0] != np.NaN:
                self.features_names = [f"{self.preset}_{i}" for i in range(1, features.shape[0] + 1)]

        return features
