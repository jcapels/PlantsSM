
import os
from deepmol.pipeline import Pipeline
from deepmol.datasets import SmilesDataset
import numpy as np

from plants_sm.pathway_prediction.pathway_classification_utils._utils import get_ec_numbers_from_ko_pathway


KEGG_PATHWAYS = np.array(['map01065',
 'map01064',
 'map01063',
 'map01070',
 'map01056',
 'map00905',
 'map00232',
 'map00906',
 'map00904',
 'map00944',
 'map00941',
 'map00901',
 'map00943',
 'map00950',
 'map00902',
 'map00940',
 'map00860',
 'map00100',
 'map00900',
 'map00960',
 'map00908'])

class PlantPathwayClassifier:

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_path = os.path.join(BASE_DIR,
                                    "pathway_prediction",
                                    "pathway_classification_utils",
                                    "kegg_pathway_prediction")
    

    def __init__(self, classification_type):
        self.classification_type = classification_type
        self.pipeline = self.load_model()

    def load_model(self):
        if self.classification_type == 'KEGG':
            return Pipeline.load(self.pipeline_path)
        else:
            raise ValueError(f"Unsupported classification type: {self.classification_type}")

    def predict(self, input_smiles):
        place_holder = "CCC" # we need to add this due to a bug in deepmol (will be fixed in next release)
        smiles_dataset = SmilesDataset(smiles=[input_smiles, place_holder], mode=["classification"]*21)
        if self.classification_type == 'KEGG':
            predictions = self.pipeline.predict(smiles_dataset)
            predictions = predictions[0]

            indices = np.where(predictions == 1)[0]

            return KEGG_PATHWAYS[indices]
        else:
            raise ValueError(f"Unsupported classification type: {self.classification_type}")
        
    def predict_ec_numbers_in_pathway(self, input_smiles):
        ecs = set()
        pathways = self.predict(input_smiles=input_smiles)
        for pathway in pathways:
            list_ecs = get_ec_numbers_from_ko_pathway(pathway)
            ecs.update(list_ecs)
        return ecs