from typing import List

import torch
from plants_sm.ml.data_structures.dataset import Dataset
from plants_sm.ml.featurization.featurizer import FeaturesGenerator
from transformers import BertModel, BertConfig
from plants_sm.ml.featurization.reactions._reaction_bert_tokenizer import ReactionSmilesTokenizer, ReactionSMILESDataset
import os
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd

class ReactionBERT(FeaturesGenerator):

    bert_model_path: str = ""
    dimension: int = 2400
    device = "cpu"


    def set_features_names(self) -> List[str]:
        """
        The method features_names will return the names of the features.
        """
        self.features_names = [f'reactionbert_{i}' for i in range(self.dimension)]
        return self.features_names
        

    def _fit(self, dataset: Dataset, instance_type: str) -> 'ReactionBERT':
        """
        Abstract method that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        """
        # get path of this file
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.tokenizer = ReactionSmilesTokenizer(vocab_file=os.path.join(file_path, 'reaction_bert_vocab.txt'))

        config = BertConfig(vocab_size=len(self.tokenizer), 
                            max_position_embeddings=512,
                            attention_probs_dropout_prob=0.1,
                            hidden_dropout_prob=0.2,
                            num_attention_heads=8,
                            num_hidden_layers=12,
                            hidden_size=2400,
                            type_vocab_size=2,
                            layer_norm_eps=1e-12,
                            hidden_act="gelu",
                            pad_token_id=0,
                            initializer_range=0.02,
                            intermediate_size=1024)

        self.bert = BertModel.from_pretrained(self.bert_model_path, config=config)
        self.bert.to(self.device)
        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'ReactionBERT':
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

    def _featurize(self, reaction_smiles: str) -> np.ndarray:
        """
        The method _featurize will generate the desired features for a given reaction SMILES

        Parameters
        ----------
        reaction_smiles: str
            reaction SMILES string

        Returns
        -------
        features_names: List[str]
            the names of the features

        features: np.ndarray
            the features
        """
        # Prepare the dataset and dataloader
        dataset_ = ReactionSMILESDataset([reaction_smiles], None, self.tokenizer)
        dataloader = DataLoader(dataset_, batch_size=1, shuffle=False)
        # Get the BERT embeddings
        for batch in dataloader:
            with torch.no_grad():
                batch = [item.to(self.device) for item in batch]
                output = self.bert(batch[0], attention_mask=batch[1])
                output = output.last_hidden_state[:, 0, :]
                return output.detach().cpu().numpy().reshape((self.dimension, ))