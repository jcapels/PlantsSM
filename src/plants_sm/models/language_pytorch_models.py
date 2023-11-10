import torch
from torch.utils.data import DataLoader, TensorDataset

from plants_sm.data_structures.dataset import Dataset, PLACEHOLDER_FIELD
from plants_sm.models.pytorch_model import PyTorchModel
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_DIMENSIONS, ESM_FUNCTIONS, ESM_LAYERS



class ESM1bModel(PyTorchModel):

    def _preprocess_data(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        tensors = []
        sequences = [(sequence_id, dataset.instances[PLACEHOLDER_FIELD][sequence_id]) for sequence_id in
                     dataset.dataframe[dataset.instances_ids_field]]
        
        esm_callable = ESM_FUNCTIONS["esm1b_t33_650M_UR50S"]

        _, alphabet = esm_callable()
        batch_converter = alphabet.get_batch_converter()

        _, _, tokens = batch_converter(sequences)

        tensors.append(tokens)

        try:
            if dataset.y is not None:
                tensors.append(torch.tensor(dataset.y, dtype=torch.float))
        except ValueError:
            pass

        dataset = TensorDataset(
            *tensors
        )

        data_loader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size
        )
        return data_loader
