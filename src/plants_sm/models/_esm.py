from joblib import Parallel, delayed
import torch
from tqdm import tqdm
from plants_sm.data_structures.dataset.dataset import Dataset
from plants_sm.data_structures.dataset.single_input_dataset import PLACEHOLDER_FIELD
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_FUNCTIONS
from plants_sm.models.lightning_model import InternalLightningModel
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class ESM(InternalLightningModel):

    def __init__(self, module, batch_size: int = 32, devices="cpu", **trainer_kwargs) -> None:

        self.model_name = module.model_name

        super().__init__(module, batch_size, devices, **trainer_kwargs)

    def _preprocess_data(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:

        tensors = []
        sequences = [(sequence_id, dataset.instances[PLACEHOLDER_FIELD][sequence_id]) for sequence_id in
                     dataset.dataframe[dataset.instances_ids_field]]
        
        esm_callable = ESM_FUNCTIONS[self.model_name]

        _, alphabet = esm_callable()
        batch_converter = alphabet.get_batch_converter()

        # _, _, tokens = batch_converter(sequences)

        batch_size = 10000  # You can adjust this based on your preferences

        # Initialize the progress bar
        progress_bar = tqdm(total=len(sequences), desc="Processing sequences", position=0, leave=True)

        # Define the function to be parallelized
        def process_batch(batch):
            _, _, tokens = batch_converter(batch)
            return tokens

        # Process sequences in parallel with a progress bar in batches
        result_tokens = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_results = Parallel(n_jobs=-1)(delayed(process_batch)(batch) for batch in [batch])
            result_tokens.extend(batch_results)
            progress_bar.update(len(batch))

        # Close the progress bar
        progress_bar.close()

        # Use joblib to parallelize the function across sequences
        result_tokens = torch.cat(result_tokens, dim=0)
        tensors.append(result_tokens)

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
    