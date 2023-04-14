from typing import Dict, Tuple

from rdkit.Chem import MolFromSmiles, MolToSmiles

from plants_sm.data_standardization.compounds._presets import DEEPMOL_STANDARDIZERS
from plants_sm.data_structures.dataset import Dataset
from plants_sm.transformation._utils import transform_instances
from plants_sm.transformation.transformer import Transformer


class DeepMolStandardizer(Transformer):
    preset: str = "custom_standardizer"
    kwargs: Dict = {}

    def _fit(self, dataset: Dataset, instance_type: str) -> 'DeepMolStandardizer':
        """
        Method to fit the transformer

        Parameters
        ----------

        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.

        Returns
        -------
        self: DeepMolStandardizer
        """
        if self.preset not in DEEPMOL_STANDARDIZERS:
            raise ValueError(f'Preset {self.preset} is not available.')

        descriptor = DEEPMOL_STANDARDIZERS[self.preset]
        self.descriptor = descriptor(**self.kwargs)
        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'DeepMolStandardizer':
        """
        Method to fit the transformer

        Parameters
        ----------

        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.

        Returns
        -------
        self: DeepMolStandardizer
        """
        return self._fit(dataset, instance_type)

    def _transform(self, dataset: Dataset, instance_type: str) -> Dataset:
        """
        Method to transform the dataset

        Parameters
        ----------
        dataset: Dataset
            dataset to be transformed

        Returns
        -------
        dataset: Dataset
            transformed dataset

        """
        return transform_instances(self.n_jobs, dataset, self._compound_preprocessing, instance_type, self.preset)

    def _compound_preprocessing(self, compound: str, identifier: str) -> Tuple[str, str]:
        """
        Method to preprocess a compound

        Parameters
        ----------

        compound: str
            compound to be preprocessed

        identifier: str
            identifier of the compound

        Returns
        -------
        dict: Dict[str, str]
            dictionary with the identifier of the compound and the preprocessed compound
        """
        mol = MolFromSmiles(compound)
        return identifier, MolToSmiles(self.descriptor._standardize(mol))
