import torch
from torch.utils.data import DataLoader, TensorDataset

from plants_sm.data_structures.dataset import Dataset, PLACEHOLDER_FIELD
from plants_sm.models.pytorch_model import PyTorchModel


class LanguagePytorchModels(PyTorchModel):

    def _preprocess_data(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        tensors = []
        sequences = [dataset.instances[PLACEHOLDER_FIELD][sequence_id] for sequence_id in
                     dataset.dataframe[dataset.instances_ids_field]]
        tensor = torch.tensor(sequences, dtype=torch.float)
        tensors.append(tensor)

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
