
import os
import webbrowser
from deepmol.pipeline import Pipeline
from deepmol.datasets import SmilesDataset
import numpy as np

from plants_sm.io.pickle import read_pickle
from plants_sm.pathway_prediction.pathway_classification_utils._utils import get_ec_numbers_from_ko_pathway
from plants_sm.pathway_prediction.solution import ECSolution


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
                            'map00908'
                            ])

class PlantPathwayClassifier:

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kegg_pipeline_path = os.path.join(BASE_DIR,
                                    "pathway_prediction",
                                    "pathway_classification_utils",
                                    "kegg_pathway_prediction")
    
    plantcyc_pipeline_path = os.path.join(BASE_DIR,
                                    "pathway_prediction",
                                    "pathway_classification_utils",
                                    "plantcyc_pathway_prediction")
    
    plantcyc_labels_path = os.path.join(BASE_DIR,
                                    "pathway_prediction",
                                    "pathway_classification_utils",
                                    "plantcyc_labels.pkl")
    
    plantcyc_pathways_to_reaction_path = os.path.join(BASE_DIR,
                                    "pathway_prediction",
                                    "pathway_classification_utils",
                                    "plantcyc_pathway_reactions.pkl")
    
    plantcyc_reaction_to_ec_path = os.path.join(BASE_DIR,
                                    "pathway_prediction",
                                    "pathway_classification_utils",
                                    "plantcyc_reactions_ec.pkl")
    
    

    def __init__(self, classification_type):
        assert classification_type in ["KEGG", "PlantCyc"]
        self.classification_type = classification_type
        self.pipeline = self.load_model()

    def load_model(self):
        if self.classification_type == 'KEGG':
            return Pipeline.load(self.kegg_pipeline_path)
        elif self.classification_type == "PlantCyc":
            return Pipeline.load(self.plantcyc_pipeline_path)

    def predict(self, input_smiles):
        place_holder = "CCC" # we need to add this due to a bug in deepmol (will be fixed in next release)
        if self.classification_type == 'KEGG':
            smiles_dataset = SmilesDataset(smiles=[input_smiles, place_holder], mode=["classification"]*21)
            predictions = self.pipeline.predict(smiles_dataset)
            predictions = predictions[0]
            indices = np.where(predictions == 1)[0]
            return KEGG_PATHWAYS[indices]
        
        elif self.classification_type == 'PlantCyc':
            PLANTCYC_PATHWAYS = np.array(read_pickle(self.plantcyc_labels_path))
            smiles_dataset = SmilesDataset(smiles=[input_smiles, place_holder], mode=["classification"]*len(PLANTCYC_PATHWAYS))
            predictions = self.pipeline.predict(smiles_dataset)
            predictions = predictions[0]
            indices = np.where(predictions == 1)[0]
            return PLANTCYC_PATHWAYS[indices]
        
    def predict_ec_numbers_in_pathway(self, input_smiles):
        ecs = set()
        pathways = self.predict(input_smiles=input_smiles)
        if self.classification_type == "KEGG":
            for pathway in pathways:
                list_ecs = get_ec_numbers_from_ko_pathway(pathway)
                ecs.update(list_ecs)
        elif self.classification_type == "PlantCyc":
            pathways_to_reaction = read_pickle(self.plantcyc_pathways_to_reaction_path)
            reaction_to_ec = read_pickle(self.plantcyc_reaction_to_ec_path)
            for pathway in pathways:
                reactions = pathways_to_reaction[pathway]
                for reaction in reactions:
                    if reaction in reaction_to_ec:
                        ecs.update(reaction_to_ec[reaction])

        return ecs, pathways

    def predict_ecs_present_in_pathways(self, smiles_input: str, ec_solution: ECSolution):
        ec_solution.create_ec_to_entities()
        predictions, pathways = self.predict_ec_numbers_in_pathway(smiles_input)

        result = {}
        for prediction in predictions:
            ec = ec_solution.get_entities(prediction)
            if ec:
                result[prediction] = ec

        for pathway in pathways:
            ecs = list(result.keys())
            ec_str = "+%20".join(ecs)
            ec_str = "%20"+ec_str
            url = f"https://www.genome.jp/kegg-bin/show_pathway?{pathway}/{ec_str}"

            webbrowser.open(url)

        return pathways