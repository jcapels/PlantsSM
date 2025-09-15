import os
from typing import Tuple
from lightning import Trainer
import numpy as np
import pandas as pd
import torch
from plants_sm.io.pickle import read_pickle
from plants_sm.pathway_prediction.annotator import Annotator
from plants_sm.pathway_prediction.entities import Reaction
from plants_sm.pathway_prediction.solution import ECSolution
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from chemprop.data import ReactionDatapoint
from chemprop.featurizers import MultiHotAtomFeaturizer
from chemprop.featurizers.molgraph.reaction import CondensedGraphOfReactionFeaturizer
from chemprop.data.datasets import ReactionDataset
from chemprop import data
from chemprop import models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ReactionECNumberAnnotator(Annotator):
    """
    Annotates reaction SMILES with EC numbers using a trained Chemprop model.

    Attributes
    ----------
    solution : ECSolution, optional
        The solution object to store annotation results.
    model_path : str
        Path to the saved Chemprop model checkpoint.
    label_encoder_path : str
        Path to the saved label encoder pickle file.
    atom_featurizer : MultiHotAtomFeaturizer
        Atom featurizer for molecular graph representation.
    rxn_featurizer : CondensedGraphOfReactionFeaturizer
        Reaction featurizer for graph-based reaction representation.
    model : models.MPNN
        The loaded Chemprop model for EC number prediction.
    trainer : Trainer
        PyTorch Lightning trainer for model inference.
    label_encoder : LabelEncoder
        Encoder for EC number labels.
    """

    solution: ECSolution = None
    model_path: str = os.path.join(
        BASE_DIR,
        "pathway_prediction",
        "reaction_annotator_utils",
        "chemprop.ckpt"
    )
    label_encoder_path: str = os.path.join(
        BASE_DIR,
        "pathway_prediction",
        "reaction_annotator_utils",
        "label_encoder.pkl"
    )

    atom_featurizer = MultiHotAtomFeaturizer.organic()
    rxn_featurizer = CondensedGraphOfReactionFeaturizer(atom_featurizer=atom_featurizer)

    def __init__(self, device: str = "cpu"):
        """
        Initialize the ReactionECNumberAnnotator.

        Parameters
        ----------
        device : str, default="cpu"
            Device to run the model on ("cpu" or "cuda").
        """
        self.model = models.MPNN.load_from_checkpoint(
            self.model_path,
            map_location=torch.device(device)
        )
        if "cuda" in device:
            self.trainer = Trainer(accelerator="gpu", devices=device)
        else:
            self.trainer = Trainer(accelerator="cpu")
        self.label_encoder = read_pickle(self.label_encoder_path)

    def validate_input(self, entities: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate input DataFrame and separate valid and invalid reaction SMILES.

        Parameters
        ----------
        entities : pd.DataFrame
            DataFrame containing a column named "rxn_smiles" with reaction SMILES.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple of (valid_entities, invalid_entities) DataFrames.

        Raises
        ------
        ValueError
            If "rxn_smiles" is not a column in the input DataFrame.
        """
        if "rxn_smiles" not in entities.columns:
            raise ValueError(
                "rxn_smiles has to be the name of the column where the reaction SMILES are."
            )
        rxn_smis = entities.rxn_smiles

        valid_entities_idx = []
        invalid_entities_idx = []
        for i, rxn_smi in tqdm(
            enumerate(rxn_smis),
            total=len(rxn_smis),
            desc="Checking reactions validity"
        ):
            try:
                _ = ReactionDatapoint.from_smi(rxn_smi, keep_h=True)
                valid_entities_idx.append(i)
            except Exception:
                invalid_entities_idx.append(i)

        valid_entities = entities.loc[valid_entities_idx, :]
        invalid_entities = entities.loc[invalid_entities_idx, :]
        return valid_entities, invalid_entities

    def _create_reaction_datapoints(self, rxn_smis: list, shuffle: bool = True):
        """
        Create reaction datapoints and a data loader for model prediction.

        Parameters
        ----------
        rxn_smis : list
            List of reaction SMILES strings.
        shuffle : bool, default=True
            Whether to shuffle the data loader.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader for the reaction datapoints.
        """
        rxn_datapoints = []
        for rxn_smi in tqdm(rxn_smis, total=len(rxn_smis)):
            try:
                datapoint = ReactionDatapoint.from_smi(rxn_smi, keep_h=True)
                rxn_datapoints.append(datapoint)
            except RuntimeError:
                continue
        dset = ReactionDataset(rxn_datapoints, featurizer=self.rxn_featurizer)
        loader = data.build_dataloader(dset, num_workers=3, shuffle=shuffle)
        return loader

    def _convert_to_readable_format(self, file: str, format: str, **kwargs):
        """
        Convert input file to a DataFrame based on the specified format.

        Parameters
        ----------
        file : str
            Path to the input file.
        format : str
            Format of the file (only "csv" is supported).
        **kwargs : dict
            Additional keyword arguments for file reading.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the reaction data.

        Raises
        ------
        ValueError
            If the format is not supported.
        """
        if format == "csv":
            return self._dataframe_from_csv(file, **kwargs)
        else:
            raise ValueError(f"Format {format} not supported. Only csv for now")

    def _annotate(self, entities: pd.DataFrame) -> ECSolution:
        """
        Annotate reaction SMILES with predicted EC numbers.

        Parameters
        ----------
        entities : pd.DataFrame
            DataFrame containing reaction SMILES in the "rxn_smiles" column.

        Returns
        -------
        ECSolution
            An ECSolution object with annotated EC numbers and Reaction objects.
        """
        rxn_smis = entities.rxn_smiles
        loader = self._create_reaction_datapoints(rxn_smis, False)
        predictions = self.trainer.predict(self.model, loader)
        reaction_ec_1 = {}
        reaction_ec_2 = {}
        reaction_ec_3 = {}
        reaction_ec_4 = {}
        reactions = {}
        predictions = torch.cat(predictions)
        predictions = predictions.squeeze()
        predictions = predictions.detach().cpu().numpy()
        predicted_classes = np.argmax(predictions, axis=-1)
        ec_numbers = self.label_encoder.inverse_transform(predicted_classes)
        for i, entity in enumerate(entities.rxn_smiles):
            ec3 = ec_numbers[i]
            reaction_ec_3[i] = [(ec3, predictions[i, predicted_classes[i]])]
            reactions[str(i)] = Reaction.from_smiles(entity)
        return ECSolution(
            entity_ec_1=reaction_ec_1,
            entity_ec_2=reaction_ec_2,
            entity_ec_3=reaction_ec_3,
            entity_ec_4=reaction_ec_4,
            entities=reactions,
        )

    def _dataframe_from_csv(self, file: str, **kwargs) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.

        Parameters
        ----------
        file : str
            Path to the CSV file.
        **kwargs : dict
            Additional keyword arguments for pandas.read_csv.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the reaction data.
        """
        return pd.read_csv(file, **kwargs)
