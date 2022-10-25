import numpy as np
from map4 import MAP4Calculator
from rdkit.Chem import MolFromSmiles

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator


class MAP4Fingerprint(FeaturesGenerator):
    name = "mp4_fingerprint"
    dimensions = 1024
    radius = 2
    is_counted = False

    def set_features_names(self):
        return [f"mp4_fingerprint_{i}" for i in range(self.dimensions)]

    def _fit(self, dataset: Dataset, instance_type: str, **kwargs) -> 'FeaturesGenerator':
        self.fingerprint = MAP4Calculator(self.dimensions, self.radius, self.is_counted, True)
        return self

    def _featurize(self, molecule: str) -> np.ndarray:
        mol = MolFromSmiles(molecule)
        map4_fingerprint = self.fingerprint.calculate(mol)
        return np.array(map4_fingerprint)