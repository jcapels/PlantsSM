from abc import abstractmethod
from typing import Dict, List, Tuple

import pandas as pd
from plants_sm.pathway_prediction.annotator import Annotator
from plants_sm.pathway_prediction.ec_numbers_annotator_utils.predictions import predict_with_model, predict_with_model_from_fasta
from plants_sm.pathway_prediction.entities import Protein
from plants_sm.pathway_prediction.solution import ECSolution


class ECAnnotator(Annotator):

    solution: ECSolution = None

    def annotate_from_fasta(self, file, **kwargs) -> ECSolution:

        proteins = Protein.from_fasta(file)

        results = self._predict_from_fasta(file, **kwargs)

        return self._annotate((proteins, results))

    @abstractmethod
    def _predict_from_fasta(self, file: str, **kwargs) -> pd.DataFrame:
        pass

    def _annotate(self, results: Tuple[Dict[str, Protein], pd.DataFrame]) -> ECSolution:

        proteins, results = results
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
                enzymes_ec_1[protein_id] = [(x.split(":")[0], float(x.split(":")[1])) for x in result["EC1"].split(";")]
            if EC2 != "":
                enzymes_ec_2[protein_id] = [(x.split(":")[0], float(x.split(":")[1])) for x in result["EC2"].split(";")]
            if EC3 != "":
                enzymes_ec_3[protein_id] = [(x.split(":")[0], float(x.split(":")[1])) for x in result["EC3"].split(";")]
            if EC4 != "":
                enzymes_ec_4[protein_id] = [(x.split(":")[0], float(x.split(":")[1])) for x in result["EC4"].split(";")]

        return ECSolution(
                entity_ec_1=enzymes_ec_1,
                entity_ec_2=enzymes_ec_2,
                entity_ec_3=enzymes_ec_3,
                entity_ec_4=enzymes_ec_4,
                entities=proteins 
            )

    
    def annotate_from_csv(self, file: str, **kwargs) -> ECSolution:
        """
        Read a CSV file and create ECSolution objects from the sequences.
        The CSV file should have columns "id" and "sequence".
        Parameters
        ----------
        file : str
            Path to the CSV file.
        **kwargs : dict
            Additional keyword arguments to pass to pandas read_csv.
        Returns
        -------
        ECSolution
            An ECSolution object containing the annotated EC numbers.
        """
        
        proteins = Protein.from_csv(file, **kwargs)

        results = self._predict_from_csv(file, **kwargs)

        return self._annotate((proteins, results))
    
    @abstractmethod
    def _predict_from_csv(self, file: str, **kwargs) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses to annotate from a CSV file.
        """
        pass

    def _annotate_from_file(self, file: str, format: str, **kwargs) -> ECSolution:
        """
        Annotate from a file based on the specified format.
        Parameters
        ----------
        file : str
            Path to the file.
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
            return self.annotate_from_fasta(file, **kwargs)
        
        elif format in ["csv", "tsv"]:
            return self.annotate_from_csv(file, **kwargs)

class ProtBertECAnnotator(ECAnnotator):

    pipeline: str = "DNN ProtBERT all data"

    def _predict_from_fasta(self, file: str, **kwargs) -> pd.DataFrame:
        """
        Predict EC numbers from a FASTA file using the ProtBERT model.
        
        Parameters
        ----------
        file : str
            Path to the FASTA file.
        **kwargs : dict
            Additional keyword arguments to pass to the fasta reading function.
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predictions.
        """

        return predict_with_model_from_fasta(self.pipeline, file, **kwargs)
    
    def _predict_from_csv(self, file: str, **kwargs) -> pd.DataFrame:
        """
        Predict EC numbers from a CSV file using the ProtBERT model.
        Parameters
        ----------
        file : str
            Path to the CSV file.
        **kwargs : dict
            Additional keyword arguments to pass to the prediction function.
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predictions.
        """
        if "sequences_field" not in kwargs:
            kwargs["sequences_field"] = "sequence" 
        if "ids_field" not in kwargs:
            kwargs["ids_field"] = "id"
        return predict_with_model(self.pipeline, file, **kwargs)