from typing import List

import pandas as pd
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_standardization.truncation import Truncator
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.pathway_prediction.annotator import Annotator
from plants_sm.pathway_prediction.solution import Solution

from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert

import torch
from torch import nn
from plants_sm.models.fc.fc import DNN
from plants_sm.models.pytorch_model import PyTorchModel

class ESIAnnotator(Annotator):

    device: str = "cpu"

    def apply_transfer_learning_to_features(dataset: MultiInputDataset):

        protein_model_ = torch.load("prot_bert.pt", map_location="cpu")
        protein_model = DNN(1024, [2560], 5743, batch_norm=True, last_sigmoid=True)

        protein_model.load_state_dict(protein_model_)
        model = PyTorchModel(model=protein_model, loss_function=nn.BCELoss, model_name="ec_number", device="cuda:3")
        
        embedding = model.get_embeddings(dataset)

        features = {"proteins": {}}

        for ids, emb in zip(dataset.identifiers, embedding):
            features["proteins"][ids] = emb

        dataset.add_features(features)

    def _annotate_from_csv(self, file: str, **kwargs) -> List[Solution]:
        header = pd.read_csv(file, nrows=1)
        if header.shape[0] < 4:
            raise ValueError("CSV file must contain at least four columns.")

        protein_ids = header.columns[0]
        protein_sequences = header.columns[1]
        compound_ids = header.columns[2]
        compound_smiles = header.columns[3]
        dataset = MultiInputDataset.from_csv(file, representation_field={"ligands": compound_smiles, "proteins": protein_sequences}, 
                                             id_field={"ligands": compound_ids, "proteins": protein_ids}, **kwargs)
        
        truncator = Truncator(max_length=884)
        protein_standardizer = ProteinStandardizer()

        multi_input_dataset = protein_standardizer.fit_transform(multi_input_dataset, "proteins")
        multi_input_dataset = truncator.fit_transform(multi_input_dataset, "proteins")
        
        transformer = ProtBert(batch_size=1, device=self.device)
        multi_input_dataset = transformer.fit_transform(multi_input_dataset, "proteins")

        from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors

        featurizer = DeepMolDescriptors(preset="np_classifier_fp", kwargs={"useChirality": False})
        featurizer.fit_transform(multi_input_dataset, "ligands")
        

    def _annotate_from_file(self, file: str, format: str, **kwargs) -> List[Solution]:
        if format == "csv":
            return self._annotate_from_csv(file, **kwargs)
        else:
            raise ValueError(f"Format {format} not supported.")
        

    