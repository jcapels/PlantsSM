import numpy as np
from transformers import AutoTokenizer, AutoModel

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.featurization.proteins.bio_embeddings._utils import get_device, reduce_per_protein


class ProtBert(FeaturesGenerator):

    embedding_dimension = 1024
    output_shape_dimension: int = 2

    def set_features_names(self):
        """
        The method features_names will set the names of the features.
        """
        self.features_names = [f"prot_bert_{num}" for num in range(1, self.embedding_dimension + 1)]

    def _fit(self, dataset: Dataset, instance_type: str) -> 'ProtBert':
        """
        Fit the ProtBert model to the dataset.

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.

        Returns
        -------
        self: Estimator
            the fitted ProtBert
        """
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = AutoModel.from_pretrained("Rostlab/prot_bert")

        device_to_use = get_device(self.device)
        self.model = model.to(device_to_use)
        self.model = model.eval()

        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'ProtBert':
        """
        Fit the ProtBert model to the dataset.

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        instance_type: str
            type of the instances to be featurized

        Returns
        -------
        self: Estimator
            the fitted ProtBert
        """
        return self._fit(dataset, instance_type)

    def _featurize(self, sequence: str) -> np.ndarray:
        """
        The method _featurize will generate the desired features for a given protein sequence

        Parameters
        ----------
        sequence: str
            instance to be featurized

        Returns
        -------
        features: np.ndarray
            features generated for the given protein sequence

        """
        sequence = ' '.join([char for char in sequence])
        encoded_input = self.tokenizer(sequence, return_tensors='pt')
        encoded_input.to(get_device(self.device))
        output = self.model(**encoded_input)
        tensor = output['last_hidden_state'].detach().cpu().numpy()
        embedding = tensor[0][1:tensor[0].shape[0] - 1, :]

        if self.output_shape_dimension == 2:
            return reduce_per_protein(embedding)

        elif self.output_shape_dimension == 3:
            return embedding
