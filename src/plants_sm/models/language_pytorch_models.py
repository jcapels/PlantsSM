from plants_sm.models.lightning_model import LightningModelModule
import torch
from torch.utils.data import DataLoader, TensorDataset

from plants_sm.data_structures.dataset import Dataset, PLACEHOLDER_FIELD
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_FUNCTIONS



class EC_ESM2LightningModel(LightningModelModule):

    def __init__(self, model, batch_size: int = 32, devices="cpu", model_name="esm2_t12_35M_UR50D", **kwargs):
        super().__init__(model, batch_size, devices, **kwargs)
        self.model_name = model_name

    def _preprocess_data(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        tensors = []
        sequences = [(sequence_id, dataset.instances[PLACEHOLDER_FIELD][sequence_id]) for sequence_id in
                     dataset.dataframe[dataset.instances_ids_field]]
        
        esm_callable = ESM_FUNCTIONS[self.model_name]

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

    
class EC_ESM1bLightningModel(LightningModelModule):

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
