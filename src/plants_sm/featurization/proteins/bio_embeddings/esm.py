from collections import ChainMap

import esm
import numpy as np
import torch

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization._utils import call_set_features_names
from plants_sm.featurization.proteins.bio_embeddings._utils import get_device
from plants_sm.transformation.transformer import Transformer


class ESM1bEncoder(Transformer):
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

    preset: str = 'representations'
    batch_size: int = 16

    def set_features_names(self):
        self.features_names = [f"esm_feature_{i}" for i in range(1, 1280 + 1)]

    def _fit(self, dataset: Dataset, instance_type: str) -> 'ESM1bEncoder':
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

        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        device = get_device()

        self.model = model.to(device)
        self.model = model.eval()
        self.batch_converter = alphabet.get_batch_converter()

        return self

    @call_set_features_names
    def _transform(self, dataset: Dataset, instance_type: str) -> Dataset:
        """
        It encodes a protein sequence with the embedding layer of the pre-trained model ESM-1B.

        Parameters
        ----------
        sequence: str
            The protein sequence to be encoded.

        Returns
        -------
        encoded_sequence: np.ndarray
            The encoded protein sequence.
        """
        # it has to run in batch of 16, otherwise can lead to OOM issues

        res = []
        batch = []
        batch_ids = []
        for instance_id, instance_representation in dataset.get_instances(instance_type).items():
            batch.append((instance_id, instance_representation))
            batch_ids.append(instance_id)
            if len(batch) == self.batch_size:
                representations = {}
                batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
                # Extract per-residue representations (on CPU)
                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)

                representations['representations'] = results["representations"][33]
                representations['features'] = results["logits"]

                sequences_embedding = representations[self.preset][0].numpy()

                temp_result = {}
                for i, batch_instance_id in enumerate(batch_ids):
                    temp_result[batch_instance_id] = sequences_embedding[i]

                res.append(temp_result)
                batch = []
                batch_ids = []

        if len(batch) != 0:
            representations = {}
            batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)

            representations['representations'] = results["representations"][33]
            representations['features'] = results["logits"]

            sequences_embedding = representations[self.preset][0].numpy()

            temp_result = {}
            for i, batch_instance_id in enumerate(batch_ids):
                temp_result[batch_instance_id] = sequences_embedding[i]

            res.append(temp_result)

        dataset.features[instance_type] = dict(ChainMap(*res))

        if instance_type not in dataset.features_fields:
            dataset.features_fields[instance_type] = self.features_names
        else:
            dataset.features_fields[instance_type].extend(self.features_names)

        return dataset
