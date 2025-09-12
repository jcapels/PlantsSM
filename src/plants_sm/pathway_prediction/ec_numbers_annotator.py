from abc import abstractmethod
import os
from typing import Tuple
import pandas as pd
from plants_sm.pathway_prediction._validation_utils import _validate_proteins
from plants_sm.pathway_prediction.annotator import Annotator
from plants_sm.pathway_prediction.ec_numbers_annotator_utils._utils import fasta_to_dataframe
from plants_sm.pathway_prediction.ec_numbers_annotator_utils.esm1b_predictions import (
    predict_with_esm1b_from_dataframe,
)
from plants_sm.pathway_prediction.ec_numbers_annotator_utils.prot_bert_prediction import (
    predict_with_protbert_from_dataframe,
)
from plants_sm.pathway_prediction.entities import Protein
from plants_sm.pathway_prediction.solution import ECSolution

class ECAnnotator(Annotator):
    """Base class for EC number annotators."""

    solution: ECSolution = None

    def _dataframe_from_fasta(self, file: str, **kwargs) -> pd.DataFrame:
        """
        Convert a FASTA file to a pandas DataFrame.

        Parameters
        ----------
        file : str
            Path to the FASTA file.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        pd.DataFrame
            DataFrame containing protein data parsed from the FASTA file.
        """
        entities = fasta_to_dataframe(file)
        return entities

    @abstractmethod
    def _make_predictions_from_dataframe(self, entities: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Make EC number predictions from a DataFrame of protein data.

        Parameters
        ----------
        entities : pd.DataFrame
            DataFrame containing protein data.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        pd.DataFrame
            DataFrame with predicted EC numbers.
        """
        pass

    def validate_input(self, entities: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate input protein data and separate valid and invalid entries.

        Parameters
        ----------
        entities : pd.DataFrame
            DataFrame containing protein data to validate.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple of (valid_entities, invalid_entities) DataFrames.
        """
        valid_entities_ids = _validate_proteins(entities)
        valid_mask = entities[entities.columns[0]].isin(valid_entities_ids)
        valid_entities = entities[valid_mask]
        # Get invalid rows
        invalid_entities = entities[~valid_mask]
        valid_entities.reset_index(inplace=True, drop=True)
        invalid_entities.reset_index(inplace=True, drop=True)
        # Return valid entities, unique proteins, and unique compounds
        return valid_entities, invalid_entities

    def _annotate(
        self, entities: pd.DataFrame, **kwargs
    ) -> ECSolution:
        """
        Annotate proteins with EC numbers based on prediction results.

        Parameters
        ----------
        entities : pd.DataFrame
            DataFrame containing protein data and prediction results.
        **kwargs : dict
            Additional keyword arguments:
                sequences_field: str
                    Path to the database.
                ids_field: str
                    Field containing the ids.
                output_path: str
                    Path to the output file.
                all_data: bool
                    Use all data from the dataset.
                device: str
                    Device to use.
                num_gpus: int
                    Number of GPUs to use for predicting the ESM embedding.

        Returns
        -------
        ECSolution
            An ECSolution object containing the annotated EC numbers for each protein.
        """
        if "sequences_field" not in kwargs:
            kwargs["sequences_field"] = "sequence"
        if "ids_field" not in kwargs:
            kwargs["ids_field"] = "id"
        proteins = Protein.from_sequences(
            entities.loc[:, kwargs["ids_field"]],
            entities.loc[:, kwargs["sequences_field"]],
        )
        results = self._make_predictions_from_dataframe(entities=entities, **kwargs)
        enzymes_ec_1 = {}
        enzymes_ec_2 = {}
        enzymes_ec_3 = {}
        enzymes_ec_4 = {}
        for _, result in results.iterrows():
            protein_id = result["accession"]
            EC1 = result["EC1"]
            EC2 = result["EC2"]
            EC3 = result["EC3"]
            EC4 = result["EC4"]

            if EC1 != "":
                enzymes_ec_1[protein_id] = [
                    (x.split(":")[0], float(x.split(":")[1])) for x in result["EC1"].split(";")
                ]
            if EC2 != "":
                enzymes_ec_2[protein_id] = [
                    (x.split(":")[0], float(x.split(":")[1])) for x in result["EC2"].split(";")
                ]
            if EC3 != "":
                enzymes_ec_3[protein_id] = [
                    (x.split(":")[0], float(x.split(":")[1])) for x in result["EC3"].split(";")
                ]
            if EC4 != "":
                enzymes_ec_4[protein_id] = [
                    (x.split(":")[0], float(x.split(":")[1])) for x in result["EC4"].split(";")
                ]
        return ECSolution(
            entity_ec_1=enzymes_ec_1,
            entity_ec_2=enzymes_ec_2,
            entity_ec_3=enzymes_ec_3,
            entity_ec_4=enzymes_ec_4,
            entities=proteins,
        )

    def _convert_to_readable_format(self, file: str, format: str, **kwargs) -> ECSolution:
        """
        Annotate proteins from a file based on the specified format.

        Parameters
        ----------
        file : str
            Path to the input file.
        format : str
            Format of the file (e.g., "fasta", "csv").
        **kwargs : dict
            Additional keyword arguments to pass to the annotation methods.

        Returns
        -------
        ECSolution
            An ECSolution object containing the annotated EC numbers.
        """
        if format in ["faa", "fasta"]:
            return self._dataframe_from_fasta(file, **kwargs)
        elif format in ["csv", "tsv"]:
            return self._dataframe_from_csv(file, **kwargs)

    def _dataframe_from_csv(self, file: str, **kwargs) -> pd.DataFrame:
        """
        Convert a CSV/TSV file to a pandas DataFrame.

        Parameters
        ----------
        file : str
            Path to the CSV/TSV file.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        pd.DataFrame
            DataFrame containing protein data parsed from the CSV/TSV file.
        """
        return pd.read_csv(file, **kwargs)

class ProtBertECAnnotator(ECAnnotator):
    """EC number annotator using ProtBERT."""

    def _make_predictions_from_dataframe(self, entities: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Make EC number predictions using ProtBERT from a DataFrame of protein data.

        Parameters
        ----------
        entities : pd.DataFrame
            DataFrame containing protein data.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        pd.DataFrame
            DataFrame with predicted EC numbers.
        """
        return predict_with_protbert_from_dataframe(entities, **kwargs)

class ESM1bECAnnotator(ECAnnotator):
    """EC number annotator using ESM-1b."""

    def _make_predictions_from_dataframe(self, entities: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Make EC number predictions using ESM-1b from a DataFrame of protein data.

        Parameters
        ----------
        entities : pd.DataFrame
            DataFrame containing protein data.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        pd.DataFrame
            DataFrame with predicted EC numbers.
        """
        return predict_with_esm1b_from_dataframe(entities, **kwargs)
