from typing import Dict

from rdkit.Chem import MolFromSmiles

from plants_sm.data_standardization.compounds._presets import DEEPMOL_STANDARDIZERS
from plants_sm.data_structures.dataset import Dataset
from plants_sm.transformation._utils import transform_instances
from plants_sm.transformation.transformer import Transformer


class DeepMolStandardizer(Transformer):
    preset: str = "custom_standardizer"

    def _fit(self, dataset: Dataset, instance_type: str, **kwargs) -> 'DeepMolStandardizer':
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
        self.descriptor = descriptor(**kwargs)
        return self

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
        return transform_instances(self.n_jobs, dataset, self._compound_preprocessing, instance_type)

    def _compound_preprocessing(self, compound: str, identifier: str) -> Dict[str, str]:
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
        return {identifier: self.descriptor._standardize(mol)}
