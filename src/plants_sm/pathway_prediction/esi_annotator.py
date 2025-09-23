from abc import abstractmethod
from copy import deepcopy
import os
from typing import List, Tuple

import pandas as pd
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_standardization.truncation import Truncator
from plants_sm.data_structures.dataset import SingleInputDataset
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
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


class ESIAnnotator(Annotator):
    
    device: str = "cpu"
    xgboost_model_path: str

    def __init__(self, device = "cpu"):
        
        self.device = device
    
    def _predict_with_xgboost(self, dataset: MultiInputDataset):
        """
        Predict using a pre-trained XGBoost model on a multi-input dataset.

        Parameters
        ----------
        dataset : MultiInputDataset
            Dataset containing features for both proteins and ligands.
            Expected to have `dataset.X["proteins"]` and `dataset.X["ligands"]` as feature matrices.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            predictions : np.ndarray
                Binary predictions (0 or 1) for each sample.
            predictions_proba : np.ndarray
                Predicted probabilities for each sample.

        Notes
        -----
        - The method concatenates protein and ligand features before prediction.
        - The XGBoost model is loaded from `self.xgboost_model_path`.
        """

        model = read_pickle(self.xgboost_model_path)
        X_all = np.concatenate([dataset.X["proteins"], dataset.X["ligands"]], axis = 1)
        X_all = xgb.DMatrix(X_all)

        predictions_proba = model.predict(X_all)
        predictions = np.round(predictions_proba)

        return predictions, predictions_proba
    
    def validate_input(self, entities: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate input DataFrame by checking protein sequences and compound SMILES.

        Parameters
        ----------
        entities : pd.DataFrame
            Input DataFrame with at least four columns:
            - Column 0: Protein IDs
            - Column 1: Protein sequences
            - Column 2: Compound IDs
            - Column 3: Compound SMILES

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            valid_entities : pd.DataFrame
                Rows with valid protein sequences and compound SMILES.
            invalid_entities : pd.DataFrame
                Rows with invalid protein sequences or compound SMILES.

        Raises
        ------
        ValueError
            If the input DataFrame has fewer than four columns.

        Notes
        -----
        - Only unique protein sequences and compound SMILES are validated.
        - Uses helper functions `_validate_proteins` and `_validate_compounds`.
        """

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
    
    @abstractmethod
    def _get_model_for_transfer_learning(self) -> PyTorchModel:
        """
        Get the model to be used for transfer learning.

        Returns
        -------
        model : PyTorchModel
            The model to be used for transfer learning.

        Notes
        -----
        - This method should be implemented in subclasses.
        """
        pass
        

    @abstractmethod
    def _transform_with_protein_language_model(self, dataset: SingleInputDataset) -> SingleInputDataset:
        """
        Transform the dataset using a protein language model to generate embeddings.

        Parameters
        ----------
        dataset : SingleInputDataset
            Dataset containing protein sequences to be transformed.

        Returns
        -------
        SingleInputDataset
            Transformed dataset with protein embeddings.

        Notes
        -----
        - This method should be implemented in subclasses.
        - The transformation typically involves standardization, truncation, and embedding generation.
        """
        pass

    def _get_features(self, dataset: SingleInputDataset):
        """
        Generate features for the given dataset using transfer learning.

        Parameters
        ----------
        dataset : SingleInputDataset
            Dataset containing protein sequences to be featurized.

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            features : dict
                Dictionary with keys "proteins" and values as another dictionary mapping protein IDs to their embeddings.
        """
        truncator = Truncator(max_length=884)
        protein_standardizer = ProteinStandardizer()

        dataset = protein_standardizer.fit_transform(dataset)
        dataset = truncator.fit_transform(dataset)

        dataset = self._transform_with_protein_language_model(dataset)

        model = self._get_model_for_transfer_learning()
        
        embedding = model.get_embeddings(dataset)

        features = {"proteins": {}}

        for ids, emb in zip(dataset.identifiers, embedding):
            features["proteins"][ids] = emb
        
        return features
        
    
    def _annotate(self, entities: pd.DataFrame) -> List[Solution]:
        """
        Annotate proteins and compounds by generating features and predicting interactions.

        Parameters
        ----------
        entities : pd.DataFrame
            Input DataFrame with at least four columns:
            - Column 0: Protein IDs
            - Column 1: Protein sequences
            - Column 2: Compound IDs
            - Column 3: Compound SMILES

        Returns
        -------
        List[Solution]
            solution : ESISolution
                Solution object containing substrate-protein predictions.

        Raises
        ------
        ValueError
            If the input is not a DataFrame or has fewer than four columns.

        Notes
        -----
        - Uses transfer learning for protein features.
        - Uses DeepMol for compound features.
        - Predicts interactions using XGBoost.
        """

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
            protein_features = self._get_features(dataset_to_generate_features)

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
            
            entities["proba"] = predictions_proba
                
            solution = ESISolution(substrate_protein_solutions = substrate_protein_solutions,
                                   dataframe_with_solutions = entities)
            return solution
                                

        else:
            raise ValueError("For now, this method only accepts pd.DataFrame as input.")
    
    def _convert_to_readable_format(self, file: str, format: str, **kwargs) -> List[Solution]:
        """
        Convert input file to a list of Solution objects.

        Parameters
        ----------
        file : str
            Path to the input file.
        format : str
            Format of the input file (e.g., "csv").
        **kwargs
            Additional keyword arguments for format-specific readers.

        Returns
        -------
        List[Solution]
            solutions : list
                List of Solution objects parsed from the input file.

        Raises
        ------
        ValueError
            If the format is not supported.

        Notes
        -----
        - Currently, only CSV format is supported.
        """
        if format == "csv":
            return self._dataframe_from_csv(file, **kwargs)
        else:
            raise ValueError(f"Format {format} not supported. Only csv for now")

class ProtBertESIAnnotator(ESIAnnotator):

    """
    This annotator takes pairs of proteins and compounds and 
    predict whether the compounds are substrates of the paired enzyme.  
    """

    device: str = "cpu"
    xgboost_model_path: str = os.path.join(BASE_DIR,
                                "pathway_prediction",
                                "esi",
                                "xgb_prot_bert_20.pkl")
    
    def _get_model_for_transfer_learning(self) -> PyTorchModel:
        """
        Get the ProtBERT model for transfer learning.

        Returns
        -------
        model : PyTorchModel
            The ProtBERT model wrapped in a PyTorchModel.

        Notes
        -----
        - Loads a pre-trained ProtBERT model from cache.
        """
        pipeline_path = _download_pipeline_to_cache("ProtBERT pipeline")

        protein_model_ = torch.load(os.path.join(pipeline_path, "prot_bert.pt"), map_location="cpu")
        protein_model = DNN(1024, [2560], 5743, batch_norm=True, last_sigmoid=True)

        protein_model.load_state_dict(protein_model_)
        model = PyTorchModel(model=protein_model, loss_function=nn.BCELoss, model_name="ec_number", device=self.device)

        return model
    
    def _transform_with_protein_language_model(self, dataset):
        """
        Transform protein sequences using a pre-trained language model.

        Parameters
        ----------
        dataset : SingleInputDataset
            Dataset containing protein sequences to be transformed.

        Returns
        -------
        SingleInputDataset
            Transformed dataset with protein embeddings.
        Notes
        -----
        - Uses ProtBERT for embedding generation.
        - Loads a pre-trained ProtBERT model from cache.
        """
        transformer = ProtBert(batch_size=1, device=self.device)
        dataset = transformer.fit_transform(dataset)
        return dataset
    

class ESM1bESIAnnotator(ESIAnnotator):

    
    """
    This annotator takes pairs of proteins and compounds and 
    predict whether the compounds are substrates of the paired enzyme.  
    """

    device: str = "cpu"
    xgboost_model_path: str = os.path.join(BASE_DIR,
                                "pathway_prediction",
                                "esi",
                                "xgb_esm1b_20.pkl")
    
    def _get_model_for_transfer_learning(self) -> PyTorchModel:
        """
        Get the ESM1b model for transfer learning.

        Returns
        -------
        model : PyTorchModel
            The ESM1b model wrapped in a PyTorchModel.

        Notes
        -----
        - Loads a pre-trained ESM1b model from cache.
        """
        pipeline_path = _download_pipeline_to_cache("ESM1b pipeline")

        protein_model_ = torch.load(os.path.join(pipeline_path, "esm1b.pt"), map_location="cpu")
        protein_model = DNN(1280, [2560, 5120], 5743, batch_norm=True, last_sigmoid=True)

        protein_model.load_state_dict(protein_model_)
        model = PyTorchModel(model=protein_model, loss_function=nn.BCELoss, model_name="ec_number", device=self.device)
        return model
    
    def _transform_with_protein_language_model(self, dataset):
        """
        Transform protein sequences using a pre-trained language model.

        Parameters
        ----------
        dataset : SingleInputDataset
            Dataset containing protein sequences to be transformed.

        Returns
        -------
        SingleInputDataset
            Transformed dataset with protein embeddings.
        Notes
        -----
        - Uses ESM1b for embedding generation.
        - Loads a pre-trained ESM1b model from cache.
        """
        transformer = ESMEncoder(esm_function="esm1b_t33_650M_UR50S", batch_size=1, device=self.device)
        dataset = transformer.fit_transform(dataset)
        return dataset
    

class ESM2ESIAnnotator(ESIAnnotator):
    """
    This annotator takes pairs of proteins and compounds and 
    predict whether the compounds are substrates of the paired enzyme.  
    """

    device: str = "cpu"
    xgboost_model_path: str = os.path.join(BASE_DIR,
                                "pathway_prediction",
                                "esi",
                                "xgb_esm2_20.pkl")
    
    def __init__(self, device="cpu", num_gpus=1):
        self.num_gpus = num_gpus
        super().__init__(device)
    
    def _get_model_for_transfer_learning(self) -> PyTorchModel:
        """
        Get the ESM2 model for transfer learning.

        Returns
        -------
        model : PyTorchModel
            The ESM2 model wrapped in a PyTorchModel.

        Notes
        -----
        - Loads a pre-trained ESM2 model from cache.
        """
        pipeline_path = _download_pipeline_to_cache("ESM2 pipeline")

        protein_model_ = torch.load(os.path.join(pipeline_path, "esm2_3b.pt"), map_location="cpu")
        protein_model = DNN(2560, [2560], 5743, batch_norm=True, last_sigmoid=True)

        protein_model.load_state_dict(protein_model_)
        model = PyTorchModel(model=protein_model, loss_function=nn.BCELoss, model_name="ec_number", device=self.device)
        return model
    
    def _transform_with_protein_language_model(self, dataset):
        """
        Transform protein sequences using a pre-trained language model.

        Parameters
        ----------
        dataset : SingleInputDataset
            Dataset containing protein sequences to be transformed.

        Returns
        -------
        SingleInputDataset
            Transformed dataset with protein embeddings.
        Notes
        -----
        - Uses ESM2 for embedding generation.
        - Loads a pre-trained ESM2 model from cache.
        """
        transformer = ESMEncoder(esm_function="esm2_t36_3B_UR50D", batch_size=1, device=self.device, num_gpus=self.num_gpus)
        dataset = transformer.fit_transform(dataset)
        return dataset