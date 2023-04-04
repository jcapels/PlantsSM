import torch
from tqdm import tqdm

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization._utils import call_set_features_names
from plants_sm.featurization.proteins.bio_embeddings._esm_utils import TorchSpawner
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_DIMENSIONS, ESM_FUNCTIONS, ESM_LAYERS
from plants_sm.transformation.transformer import Transformer


class ESMEncoder(Transformer):
    """
    It encodes protein sequences with the embedding layer of the pre-trained model ESM-1B.
    The Esm1bEncoder operates only over pandas DataFrame.

    Parameters
    ----------
    batch_size: int, optional (default=16)
        The batch size to be used in the encoding process. Higher batch sizes can lead to OOM issues.

    """

    batch_size: int = 16
    features_names = list = []
    esm_function: str = "esm2_t6_8M_UR50D"
    device: str = "cpu"
    num_gpus: int = None 
    output_dim: int = 2
    return_contacts: bool = False

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

            self.model = ESM_FUNCTIONS[self.esm_function]
            self.layers = ESM_LAYERS[self.esm_function]

            if self.num_gpus is not None:
                self.is_ddf = True
            else:
                self.num_gpus = 0
                self.is_ddf = False

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

            batch.append((instance_id, instance_representation))
            batch_ids.append(instance_id)
            if len(batch) == self.batch_size:
                representations = {}
                results = TorchSpawner.generate_esm(batch, self.layers, esm_model = self.model, num_gpus = self.num_gpus, is_ddf=self.is_ddf)
                representations['representations'] = results["representations"][self.layers].cpu().detach().numpy()

                for i, batch_instance_id in enumerate(batch_ids):
                    if self.output_dim == 2:
                        res.append((batch_instance_id,
                                representations['representations'][i, 1: len(batch[i][1]) + 1].mean(0)))
                    else:
                        res.append((batch_instance_id,
                                representations['representations'][i, 1: len(batch[i][1]) + 1]))

                batch = []
                batch_ids = []
                pbar.update(self.batch_size)

        if len(batch) != 0:

            representations = {}
            results = TorchSpawner.generate_esm(batch, self.layers, esm_model = self.model, num_gpus = self.num_gpus, is_ddf=self.is_ddf)
            representations['representations'] = results["representations"][self.layers].cpu().detach().numpy()

            for i, batch_instance_id in enumerate(batch_ids):
                if self.output_dim == 2:
                    res.append((batch_instance_id,
                            representations['representations'][i, 1: len(batch[i][1]) + 1].mean(0)))
                else:
                    res.append((batch_instance_id,
                            representations['representations'][i, 1: len(batch[i][1]) + 1]))

            batch = []
            batch_ids = []

        dataset.features[instance_type] = dict(res)

        if instance_type not in dataset.features_fields:
            dataset.features_fields[instance_type] = self.features_names
        else:
            dataset.features_fields[instance_type].extend(self.features_names)

        return dataset
