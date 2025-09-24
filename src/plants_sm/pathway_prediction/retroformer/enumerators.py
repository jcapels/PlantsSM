from enum import Enum
import os
import zipfile

import requests
from tqdm import tqdm

class RetroformerModelsDownloadPaths(Enum):

    RETROFORMER_MODEL = "https://zenodo.org/records/17194698/files/retroformer.zip?download=1"

def _download_retroformer_to_cache() -> str:
    """
    Downloads a retroformer model to the cache folder.

    Returns
    -------
    str
        Path to the downloaded pipeline.
    """
    model = RetroformerModelsDownloadPaths.RETROFORMER_MODEL.value

    pipelines_cache_path = os.path.join(os.path.expanduser("~"), ".ec_number_prediction", "pipelines")
    retroformer_zip = os.path.join(pipelines_cache_path, "retroformer.zip")
    retroformer_model = os.path.join(pipelines_cache_path, "retroformer.pt")

    if os.path.exists(retroformer_model):
        print(f"Retroformer model already in cache.")
        return retroformer_model

    print(f"Downloading retroformer model to cache...")
    response = requests.get(model, stream=True)
    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(retroformer_zip, "wb") as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

    # unzip pipeline
    print(f"Unzipping retroformer model to cache...")
    with zipfile.ZipFile(retroformer_zip, "r") as zip_ref:
        zip_ref.extractall(pipelines_cache_path)

    os.remove(retroformer_zip)

    return retroformer_model