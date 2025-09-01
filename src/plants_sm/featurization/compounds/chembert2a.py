
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.data_structures.dataset import Dataset

from typing import List

import numpy as np

import torch

from transformers import AutoModelForMaskedLM, AutoTokenizer


class ChemBert2a(FeaturesGenerator):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_features_names(self) -> List[str]:
        """
        The method features_names will return the names of the features.
        """
        self.features_names = [f"chembert2a_{i}" for i in range(600)]
        
        return self.features_names
    
    def _fit(self, dataset: Dataset, instance_type: str) -> 'ChemBert2a':
        """
        Fit method

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        instance_type: str
            type of the instances to be featurized

        Returns
        -------
        self: KPGP
        """
        self.chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        self.chemberta.to(self.device)

        return self
    
    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'ChemBert2a':
        """
        Abstract method that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        instance_type: str
            type of the instances to be featurized
        """
        return self._fit(dataset, instance_type)
    
    def _featurize(self, compound_smiles: str) -> np.ndarray:
        """
        The method _featurize will generate the desired features for a given protein sequence

        Parameters
        ----------
        protein_sequence: str
            protein sequence string

        Returns
        -------
        features_names: List[str]
            the names of the features

        features: np.ndarray
            the features
        """

        with torch.no_grad():
            padding=True
            encoded_input = self.tokenizer(compound_smiles, return_tensors="pt",padding=padding,truncation=True).to(self.device)
            model_output = self.chemberta(**encoded_input)
            
            embedding = model_output[0][:,0,:]
            embeddings_cls = embedding

            embedding = torch.mean(model_output[0],1)
            embeddings_mean = embedding

            return embeddings_mean.cpu().numpy().flatten()
