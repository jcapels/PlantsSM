from copy import deepcopy
import os
from typing import List, Tuple

import pandas as pd
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_standardization.truncation import Truncator
from plants_sm.data_structures.dataset import SingleInputDataset
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.io.pickle import read_pickle
from plants_sm.pathway_prediction._validation_utils import _validate_compounds, _validate_proteins
from plants_sm.pathway_prediction.annotator import Annotator
from plants_sm.pathway_prediction.ec_numbers_annotator_utils._utils import _download_pipeline_to_cache
from plants_sm.pathway_prediction.solution import ESISolution, Solution

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
        X_all = xgb.DMatrix(X_all)

        predictions_proba = model.predict(X_all)
        predictions = np.round(predictions_proba)

        return predictions, predictions_proba
    
    def validate_input(self, entities: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        header = entities.columns
        if len(header) < 4:
            raise ValueError("CSV file must contain at least four columns.")

        protein_ids = header[0]
        protein_sequences = header[1]
        compound_ids = header[2]
        compound_smiles = header[3]

        # Get unique protein sequences and compound SMILES
        unique_proteins = entities[[protein_ids, protein_sequences]].drop_duplicates()
        unique_compounds = entities[[compound_ids, compound_smiles]].drop_duplicates()

        # Validate unique protein sequences
        valid_protein_ids = _validate_proteins(unique_proteins)

        # Validate unique compound SMILES
        valid_compound_ids = _validate_compounds(unique_compounds)

        # Filter the original DataFrame for valid rows
        valid_mask = (
            entities[protein_ids].isin(valid_protein_ids) &
            entities[compound_ids].isin(valid_compound_ids)
        )
        valid_entities = entities[valid_mask]

        # Get invalid rows
        invalid_entities = entities[~valid_mask]

        valid_entities.reset_index(inplace=True, drop=True)
        invalid_entities.reset_index(inplace=True, drop=True)

        # Return valid entities, unique proteins, and unique compounds
        return valid_entities, invalid_entities

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
    
    def _annotate(self, entities: pd.DataFrame) -> List[Solution]:

        if isinstance(entities, pd.DataFrame):
            header = entities.columns
            if len(header) < 4:
                raise ValueError("CSV file must contain at least four columns.")
            
            protein_ids = header[0]
            protein_sequences = header[1]
            compound_ids = header[2]
            compound_smiles = header[3]

            df_copy = deepcopy(entities)

            dataset_to_generate_features = SingleInputDataset(entities, representation_field=protein_sequences, instances_ids_field=protein_ids)
            protein_features = self._apply_transfer_learning_to_features(dataset_to_generate_features)

            del dataset_to_generate_features

            dataset = MultiInputDataset(df_copy, representation_field={"ligands": compound_smiles, "proteins": protein_sequences}, 
                                                instances_ids_field={"ligands": compound_ids, "proteins": protein_ids})
            
            dataset.add_features("proteins", protein_features)

            from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors

            featurizer = DeepMolDescriptors(preset="np_classifier_fp")
            featurizer.fit_transform(dataset, "ligands")

            _, predictions_proba = self._predict_with_xgboost(dataset)

            substrate_protein_solutions = {}
            
            compound_ids = dataset.dataframe.loc[:, dataset.instances_ids_field["ligands"]]
            protein_ids = dataset.dataframe.loc[:, dataset.instances_ids_field["proteins"]]
            unique_compound_ids = pd.unique(compound_ids)

            for unique_compound in unique_compound_ids:
                substrate_protein_solutions[unique_compound] = []

            for i, proba in enumerate(predictions_proba):
                compound_id = compound_ids[i]
                protein_id = protein_ids[i]

                substrate_protein_solutions[compound_id].append((protein_id, proba))
                
            solution = ESISolution(substrate_protein_solutions = substrate_protein_solutions)
            return solution
                                

        else:
            raise ValueError("For now, this method only accepts pd.Dataframes as input.")

    def _convert_to_readable_format(self, file: str, format: str, **kwargs) -> List[Solution]:
        if format == "csv":
            return self._dataframe_from_csv(file, **kwargs)
        else:
            raise ValueError(f"Format {format} not supported. Only csv for now")
        

    