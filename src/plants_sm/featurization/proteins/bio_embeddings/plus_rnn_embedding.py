from typing import Any, Union, List, Generator, Dict

import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import DataLoader

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.featurization.proteins.bio_embeddings._utils import get_model_file, reduce_per_protein, get_device
from plants_sm.featurization.proteins.bio_embeddings.plus.config import ModelConfig, RunConfig
from plants_sm.featurization.proteins.bio_embeddings.plus.data.alphabets import Protein
from plants_sm.featurization.proteins.bio_embeddings.plus.data.dataset import Embedding_dataset, \
    collate_sequences_for_embedding
from plants_sm.featurization.proteins.bio_embeddings.plus.model.plus_rnn import PLUS_RNN, get_embedding
from plants_sm.featurization.proteins.bio_embeddings.plus.train import Trainer
from plants_sm.featurization.proteins.bio_embeddings.plus.utils import set_seeds


class PlusRNNEmbedding(FeaturesGenerator):
    """PLUS RNN Embedder
    S. Min, S. Park, S. Kim, H. -S. Choi, B. Lee and S. Yoon, "Pre-Training of Deep Bidirectional
    Protein Sequence Representations With Structural Information,"
    in IEEE Access, vol. 9, pp. 123912-123926, 2021, doi: 10.1109/ACCESS.2021.3110269.
    """
    name = "plus_rnn"
    number_of_layers = 1
    embedding_dimension = 1024

    necessary_files = ["model_file"]

    _alphabet: Protein
    _model: PLUS_RNN
    _model_cfg: ModelConfig
    _run_cfg: RunConfig
    _options: Dict[str, Any] = {}
    _model_file = None

    output_shape_dimension: int = 2

    def _fit(self, dataset: Dataset) -> 'Estimator':
        set_seeds(2020)

        # We inlined the config json files since they aren't shipped with the package
        self._alphabet = Protein()
        self._model_cfg = ModelConfig(input_dim=len(self._alphabet))
        self._model_cfg.model_type = "RNN"
        self._model_cfg.rnn_type = "B"
        self._model_cfg.num_layers = 3
        self._model_cfg.hidden_dim = 512
        self._model_cfg.embedding_dim = 100
        self._run_cfg = RunConfig(sanity_check=True)
        self._run_cfg.batch_size_eval = 512
        self._model = PLUS_RNN(self._model_cfg)

        self._model_file = self._options.get("model_file", None)
        if self._model_file is None:
            file_path = get_model_file(self.name, self.necessary_files[0])
            self._model_file = file_path

        self._model.load_weights(self._model_file)
        self._model = self._model.to(get_device(self.device))

        self.features_names = [f"{self.name}_{num}" for num in range(1, self.embedding_dimension + 1)]

        return self

    def _featurize(self, sequence: str) -> np.ndarray:
        [embedding] = self._embed_batch([sequence])
        if self.output_shape_dimension == 2:
            embedding = reduce_per_protein(embedding)
            return embedding

        elif self.output_shape_dimension == 3:

            return embedding

    def _embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        sequences = [
            self._alphabet.encode(sequence.encode().upper()) for sequence in batch
        ]
        test_dataset = [torch.from_numpy(sequence).long() for sequence in sequences]
        test_dataset = Embedding_dataset(
            test_dataset, self._alphabet, self._run_cfg, True
        )

        iterator_test = DataLoader(
            test_dataset,
            self._run_cfg.batch_size_eval,
            collate_fn=collate_sequences_for_embedding,
        )

        model_list = [self._model, "", True, False, False]
        tasks_list = [["", [], []]]  # list of lists [idx, metrics_train, metrics_eval]
        trainer = Trainer([model_list], get_embedding, self._run_cfg, tasks_list)
        for tokens, lengths in iterator_test:
            # https://github.com/pytorch/pytorch/issues/43227
            batch = (tokens.to(self.device), lengths)
            trainer.embed(batch, {"data_parallel": False})

        embeddings = trainer.tasks_dict["results_eval"][0]["embeddings"]
        # 1 is d_h with 1024 dimensions
        for i in range(len(embeddings[0])):
            yield embeddings[1][i].numpy()

        trainer.reset()
