import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import requests
import zipfile
import os

from Bio import SeqIO
import csv

from plants_sm.pathway_prediction.ec_numbers_annotator_utils.enumerators import BLASTDownloadPaths, ModelsDownloadPaths


import os
import numpy as np
import pandas as pd

from plants_sm.data_structures.dataset.dataset import Dataset
from plants_sm.io.pickle import read_pickle
from plants_sm.models._utils import multi_label_binarize
from plants_sm.pathway_prediction.ec_numbers_annotator_utils.predictions import _generate_ec_number_from_model_predictions
from plants_sm.pipeline.pipeline import Pipeline

from plants_sm.models.fc.fc import DNN
from plants_sm.models.pytorch_model import PyTorchModel

from plants_sm.pathway_prediction.ec_numbers_annotator_utils import SRC_PATH

def _make_predictions_with_model(dataset: Dataset, pipeline: Pipeline, device: str, all_data: bool = True,
                                 num_gpus: int = 1) \
        -> pd.DataFrame:
    """
    Make predictions with a model.

    Parameters
    ----------
    dataset: Dataset
        Dataset.
    pipeline: Pipeline
        Pipeline.
    device: str
        Device to use.
    all_data: bool
        Use all data from the dataset.
    num_gpus: int
        Number of GPUs to use.

    Returns
    -------
    results_dataframe: pd.DataFrame
        Results of the prediction.
    """
    pipeline.steps["place_holder"][-1].device = device
    
    if pipeline.steps["place_holder"][-1].__class__.__name__ == "ProtBert":
        pipeline.steps["place_holder"][-1].model.to(device)
        
    elif "cuda" in device:
        if device == "cuda":
            pipeline.steps["place_holder"][-1].num_gpus = num_gpus
        else:
            pipeline.steps["place_holder"][-1].num_gpus = 1

        pipeline.steps["place_holder"][-1].is_ddf = True

    for i in range(len(pipeline.models)):
        if isinstance(pipeline.models[i], PyTorchModel):
            pipeline.models[i].model.to(device)
            pipeline.models[i].device = device

    predictions = pipeline.predict(dataset, "enzyme_discrimination")
    enzymes_non_enzymes = predictions.reshape((predictions.shape[0],))

    dataset.select(dataset.identifiers[enzymes_non_enzymes==1])

    predictions_proba = pipeline.predict_proba(dataset, "ec_number", force_transform=False)
    if all_data:
        path = os.path.join(SRC_PATH, "labels_names_all_data.pkl")
    else:
        path = os.path.join(SRC_PATH, "labels_names.pkl")

    results_dataframe = pd.DataFrame(columns=["accession", "EC1", "EC2", "EC3", "EC4"])
    labels_names = read_pickle(path)
    # get all the column indexes where the value is 1
    y_pred = multi_label_binarize(predictions_proba)

    indices = [np.where(row == 1)[0].tolist() for row in y_pred]
    labels_names = np.array(labels_names)

    ids = dataset.dataframe[dataset.instances_ids_field]
    for i in range(len(indices)):
        label_predictions = labels_names[indices[i]]
        labels_proba = predictions_proba[i, indices[i]]

        EC1, EC2, EC3, EC4 = _generate_ec_number_from_model_predictions(label_predictions, labels_proba)
        label_predictions = [";".join(EC1)] + [";".join(EC2)] + [";".join(EC3)] + [";".join(EC4)]
        results_dataframe.loc[i] = [ids[i]] + label_predictions

    return results_dataframe