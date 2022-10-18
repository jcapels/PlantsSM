import numpy as np
from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit.ML.Descriptors import MoleculeDescriptors

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.compounds._presets import DEEPMOL_PRESETS
from plants_sm.featurization.featurizer import FeaturesGenerator


class DeepMolDescriptors(FeaturesGenerator):

    preset: str = "morgan_fingerprints"

    def set_features_names(self):
        """
        Method to set the names of the features
        """
        if self.preset == "2d_descriptors":
            calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            header = calc.GetDescriptorNames()
            self.features_names = header
        else:
            self.features_names = []

    def _fit(self, dataset: Dataset, **kwargs) -> 'DeepMolDescriptors':
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
        self.descriptor = descriptor(**kwargs)
        return self

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
