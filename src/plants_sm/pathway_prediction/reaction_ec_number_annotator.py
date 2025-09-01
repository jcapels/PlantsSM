

import os

from lightning import Trainer
import numpy as np
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

from chemprop.data.datasets import (
    ReactionDataset
    )
from chemprop import data
from chemprop import models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ReactionECNumberAnnotator(Annotator):

    solution: ECSolution = None
    model_path: str = os.path.join(BASE_DIR,
                                "pathway_prediction",
                                "reaction_annotator_utils",
                                "chemprop.ckpt")
    
    label_encoder_path: str = os.path.join(BASE_DIR,
                                "pathway_prediction",
                                "reaction_annotator_utils",
                                "label_encoder.pkl")

   

    atom_featurizer = MultiHotAtomFeaturizer.organic()
    rxn_featurizer = CondensedGraphOfReactionFeaturizer(atom_featurizer=atom_featurizer)

    def __init__(self, device="cpu"):
        self.model = models.MPNN.load_from_checkpoint(self.model_path, map_location=torch.device(device))
        if "cuda" in device:
            self.trainer = Trainer(accelerator="gpu", devices=device)
        else:
            self.trainer = Trainer(accelerator="cpu")
        self.label_encoder = read_pickle(self.label_encoder_path)

    def _create_reaction_datapoints(self, rxn_smis, shuffle=True):
        """
        Encodes labels and creates reaction datapoints from SMILES and labels.

        Parameters:
        rxn_smis (list): List of reaction SMILES strings.
        ys (list): List of labels corresponding to the reaction SMILES.

        Returns:
        tuple: A tuple containing a list of ReactionDatapoint objects and the fitted LabelEncoder.
        """
        # Encode labels
        atom_featurizer = MultiHotAtomFeaturizer.organic()
        rxn_featurizer = CondensedGraphOfReactionFeaturizer(atom_featurizer=atom_featurizer)

        rxn_datapoints = []

        # Create ReactionDatapoint objects
        for rxn_smi in tqdm(rxn_smis, total=len(rxn_smis)):
            try:
                datapoint = ReactionDatapoint.from_smi(rxn_smi, keep_h=True)
                rxn_datapoints.append(datapoint)
            except RuntimeError:
                continue

        dset = ReactionDataset(rxn_datapoints, featurizer=rxn_featurizer)
        loader = data.build_dataloader(dset, num_workers=3, shuffle=shuffle)

        return loader
    
    def _annotate_from_file(self, file, format, **kwargs):
        pass

    def _annotate(self, entities):
        
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
        # Assuming you have a tensor like this

        # Convert the tensor to a NumPy array
        predictions = predictions.detach().cpu().numpy()

        predicted_classes = np.argmax(predictions, axis=-1)

        ec_numbers = self.label_encoder.inverse_transform(predicted_classes)
        for i, entity in enumerate(entities.rxn_smiles):
            ec3 = ec_numbers[i]
            reaction_ec_3[i] = [(ec3,predictions[i, predicted_classes[i]])]
            reactions[str(i)] = Reaction.from_smiles(entity)

        return ECSolution(
                entity_ec_1=reaction_ec_1,
                entity_ec_2=reaction_ec_2,
                entity_ec_3=reaction_ec_3,
                entity_ec_4=reaction_ec_4,
                entities=reactions 
            )


        


