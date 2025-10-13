import os
import re
from Bio.KEGG import REST as kegg_api

from enum import Enum

from plants_sm.pathway_prediction.ec_numbers_annotator_utils._utils import _download_and_unzip_file_to_cache

class ModelsDownloadPaths(Enum):

    PLANTCYC = "https://zenodo.org/records/17315255/files/plantcyc_pathway_prediction.zip?download=1"
    KEGG = "https://zenodo.org/records/17315255/files/kegg_pathway_prediction.zip?download=1"

def get_model_path(classification_model):

    if classification_model == "PlantCyc":
        pipeline = "plantcyc_pathway_prediction"

        pipeline_url = ModelsDownloadPaths.PLANTCYC.value

    elif classification_model == "KEGG":
        pipeline = "kegg_pathway_prediction"

        pipeline_url = ModelsDownloadPaths.KEGG.value
    
    else:

        raise ValueError("This model is not available - only KEGG and PlantCyc")

    pipeline_cache_path = os.path.join(os.path.expanduser("~"), ".ec_number_prediction", "pipelines")
    pipeline_name_for_path = pipeline.replace(" ", "_")

    if os.path.exists(os.path.join(pipeline_cache_path, pipeline_name_for_path)):
        print(f"Pipeline {pipeline} already in cache.")
        return os.path.join(pipeline_cache_path, pipeline_name_for_path)

    return _download_and_unzip_file_to_cache(pipeline_url, pipeline_cache_path, pipeline)


def get_ec_numbers_from_ko_pathway(ko_pathway_id):
    ko_pathway_id = "ko" + ko_pathway_id[3:]
    # Fetch the KO pathway file
    ko_pathway_data = kegg_api.kegg_get(ko_pathway_id).read().split('\n')

    ec_numbers = set()
    in_orthology_section = False

    for line in ko_pathway_data:
        if line.startswith('ORTHOLOGY'):
            in_orthology_section = True
            continue
        if in_orthology_section and line.startswith('///'):
            break
        if in_orthology_section and '[EC:' in line:
            # Extract all EC numbers from the line
            ec_part = line.split('[EC:')[1:]
            for part in ec_part:
                ec = part.split(']')[0]
                ec_numbers.update(ec.split())

    return sorted(ec_numbers)

def get_reactions_by_ec(ec_number):
    from Bio.KEGG import REST

    # Step 1: Get reaction IDs
    link_handle = REST.kegg_link("reaction", f"ec:{ec_number}")
    
    pattern = r"R\d+"
    reaction_ids = re.findall(pattern, link_handle.read())
    
    return reaction_ids