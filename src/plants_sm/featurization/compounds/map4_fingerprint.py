import numpy as np
from map4 import MAP4Calculator
from rdkit.Chem import MolFromSmiles

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator


class MAP4Fingerprint(FeaturesGenerator):
    dimensions = 1024
    radius = 2
    is_counted = False

    def set_features_names(self):
        self.features_names = [f"map4_fingerprint_{i}" for i in range(self.dimensions)]

    def _fit(self, dataset: Dataset, instance_type: str) -> 'FeaturesGenerator':
        self.fingerprint = MAP4Calculator(self.dimensions, self.radius, self.is_counted, True)
        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'FeaturesGenerator':
        return self._fit(dataset, instance_type)

    def _featurize(self, molecule: str) -> np.ndarray:
        mol = MolFromSmiles(molecule)
        map4_fingerprint = self.fingerprint.calculate(mol)
        map4_fingerprint = map4_fingerprint.astype("int8")
        try:
            assert map4_fingerprint.shape == (self.dimensions,)
        except AssertionError:
            map4_fingerprint = np.zeros(self.dimensions, dtype=np.int8)
        return map4_fingerprint
