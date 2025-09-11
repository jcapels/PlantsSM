import os
from typing import List

import pandas as pd
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_standardization.truncation import Truncator
from plants_sm.data_structures.dataset import SingleInputDataset
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.io.pickle import read_pickle
from plants_sm.pathway_prediction.annotator import Annotator
from plants_sm.pathway_prediction.ec_numbers_annotator_utils._utils import _download_pipeline_to_cache
from plants_sm.pathway_prediction.solution import Solution

from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert

import torch
from torch import nn
from plants_sm.models.fc.fc import DNN
from plants_sm.models.pytorch_model import PyTorchModel

import xgboost as xgb

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ProtBertESIAnnotator(Annotator):

    device: str = "cpu"
    xgboost_model_path: str = os.path.join(BASE_DIR,
                                "pathway_prediction",
                                "esi",
                                "xgb_prot_bert_20.pkl")
    
    def _predict_with_xgboost(self, dataset: MultiInputDataset):
        model = read_pickle(self.xgboost_model_path)
        X_all = np.concatenate([dataset.X["proteins"], dataset.X["ligands"]], axis = 1)
        dataset = xgb.DMatrix(X_all)

        predictions_proba = model.predict(X_all)
        predictions = np.round(predictions_proba)

        return predictions
    

    def _apply_transfer_learning_to_features(self, dataset: SingleInputDataset):

        truncator = Truncator(max_length=884)
        protein_standardizer = ProteinStandardizer()

        dataset = protein_standardizer.fit_transform(dataset)
        dataset = truncator.fit_transform(dataset)
        
        transformer = ProtBert(batch_size=1, device=self.device)
        dataset = transformer.fit_transform(dataset)

        pipeline_path = _download_pipeline_to_cache("ProtBERT pipeline")

        protein_model_ = torch.load(os.path.join(pipeline_path, "prot_bert.pt"), map_location="cpu")
        protein_model = DNN(1024, [2560], 5743, batch_norm=True, last_sigmoid=True)

        protein_model.load_state_dict(protein_model_)
        model = PyTorchModel(model=protein_model, loss_function=nn.BCELoss, model_name="ec_number", device=self.device)

        embedding = model.get_embeddings(dataset)

        features = {"proteins": {}}

        for ids, emb in zip(dataset.identifiers, embedding):
            features["proteins"][ids] = emb
        
        return features

    def annotate_from_csv(self, file: str, **kwargs) -> List[Solution]:
        header = pd.read_csv(file, nrows=1)
        if header.shape[0] < 4:
            raise ValueError("CSV file must contain at least four columns.")

        protein_ids = header.columns[0]
        protein_sequences = header.columns[1]
        compound_ids = header.columns[2]
        compound_smiles = header.columns[3]

        dataset_to_generate_features = SingleInputDataset.from_csv(dataframe=file, representation_field=protein_sequences, instances_ids_field=protein_ids)
        protein_features = self._apply_transfer_learning_to_features(dataset_to_generate_features)

        del dataset_to_generate_features

        dataset = MultiInputDataset.from_csv(file, representation_field={"ligands": compound_smiles, "proteins": protein_sequences}, 
                                             id_field={"ligands": compound_ids, "proteins": protein_ids}, **kwargs)
        
        dataset.add_features("proteins", protein_features)

        from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors

        featurizer = DeepMolDescriptors(preset="np_classifier_fp", kwargs={"useChirality": False})
        featurizer.fit_transform(dataset, "ligands")

        predictions = self._predict_with_xgboost(dataset)


    def _annotate_from_file(self, file: str, format: str, **kwargs) -> List[Solution]:
        if format == "csv":
            return self.annotate_from_csv(file, **kwargs)
        else:
            raise ValueError(f"Format {format} not supported.")
        

    