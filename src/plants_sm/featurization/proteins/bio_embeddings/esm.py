import numpy as np
import torch
from tqdm import tqdm

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization._utils import call_set_features_names
from plants_sm.featurization.proteins.bio_embeddings._utils import get_device
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_DIMENSIONS, ESM_FUNCTIONS, ESM_LAYERS
from plants_sm.transformation.transformer import Transformer


class ESMEncoder(Transformer):
    """
    It encodes protein sequences with the embedding layer of the pre-trained model ESM-1B.
    The Esm1bEncoder operates only over pandas DataFrame.

    Parameters
    ----------
    preset: str, optional (default='representations')
        The possible output from esm1b model. Available presets are:
        - 'representations': Preset to obtain the transformer representation of the protein sequences. This preset
        creats a 1280 x length of sequence matrix as an output for each sequence. More information can be accessed in
        https://github.com/facebookresearch/esm
        - 'features': Preset to obtain the input features to the transformers layers. This preset creates a
        33 x length of sequence matrix as an output for each sequence.
    batch_size: int, optional (default=16)
        The batch size to be used in the encoding process. Higher batch sizes can lead to OOM issues.

    """

    batch_size: int = 16
    features_names = list = []
    esm_function: str = "esm2_t6_8M_UR50D"
    device: str = "cpu"

    def set_features_names(self):
        self.features_names = [f"ESM_{i}" for i in range(ESM_DIMENSIONS[self.esm_function])]

    def _fit(self, dataset: Dataset, instance_type: str) -> 'ESMEncoder':
        """
        Fit the Esm1bEncoder. It loads the pre-trained model and the batch converter.

        Parameters
        ----------
        dataset: Dataset
            The dataset to be used to fit the Esm1bEncoder.

        Returns
        -------
        encoder: a fitted Esm1bEncoder
        """

        if self.esm_function in ESM_DIMENSIONS:

            self.esm_callable = ESM_FUNCTIONS[self.esm_function]

            model, alphabet = self.esm_callable()

            self.model = model.to(self.device)
            self.model = model.eval()
            self.batch_converter = alphabet.get_batch_converter()

            self.layers = ESM_LAYERS[self.esm_function]

            return self
        else:
            raise ValueError(f"Invalid esm_function. Available functions are: {list(ESM_DIMENSIONS.keys())}")

    @call_set_features_names
    def _transform(self, dataset: Dataset, instance_type: str) -> Dataset:
        """
        It encodes a protein sequence with the embedding layer of the pre-trained model ESM-1B.

        Parameters
        ----------
        dataset: Dataset
            The dataset to be used to encode the protein sequences.
        instance_type: str
            The instance type to be encoded.

        Returns
        -------
        encoded_sequence: np.ndarray
            The encoded protein sequence.
        """
        # it has to run in batch of 16, otherwise can lead to OOM issues

        res = []
        batch = []
        batch_ids = []
        instances = dataset.get_instances(instance_type)
        pbar = tqdm(desc="ESM", total=len(instances.items()))
        for instance_id, instance_representation in instances.items():
            if len(instance_representation) <= 1024:
                batch.append((instance_id, instance_representation))
            else:
                batch.append((instance_id, instance_representation[:1024]))
            batch_ids.append(instance_id)
            if len(batch) == self.batch_size:
                representations = {}
                batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
                # Extract per-residue representations (on CPU)
                # with torch.no_grad():
                batch_tokens = batch_tokens.to(self.device)
                results = self.model(batch_tokens, repr_layers=[self.layers], return_contacts=True)

                representations['representations'] = results["representations"][self.layers].cpu().detach().numpy()

                for i, batch_instance_id in enumerate(batch_ids):
                    res.append((batch_instance_id,
                                representations['representations'][i, 1: len(batch[i][1]) + 1].mean(0)))

                batch = []
                batch_ids = []
                pbar.update(self.batch_size)

        if len(batch) != 0:
            representations = {}
            batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
            # Extract per-residue representations (on CPU)
            # with torch.no_grad():
            batch_tokens = batch_tokens.to(self.device)
            results = self.model(batch_tokens, repr_layers=[self.layers], return_contacts=True)

            representations['representations'] = results["representations"][self.layers].cpu().detach().numpy()

            for i, batch_instance_id in enumerate(batch_ids):
                res.append((batch_instance_id, representations['representations'][i, 1: len(batch[i][1]) + 1].mean(0)))

        dataset.features[instance_type] = dict(res)

        if instance_type not in dataset.features_fields:
            dataset.features_fields[instance_type] = self.features_names
        else:
            dataset.features_fields[instance_type].extend(self.features_names)

        return dataset
