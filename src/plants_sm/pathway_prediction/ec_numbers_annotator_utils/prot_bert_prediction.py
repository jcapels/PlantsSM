

import os
import numpy as np
import pandas as pd

from plants_sm.data_structures.dataset.dataset import Dataset
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
from plants_sm.io.pickle import read_pickle
from plants_sm.models._utils import multi_label_binarize
from plants_sm.pathway_prediction.ec_numbers_annotator_utils._utils import _download_pipeline_to_cache, convert_fasta_to_csv
from plants_sm.pathway_prediction.ec_numbers_annotator_utils.predictions import _generate_ec_number_from_model_predictions
from plants_sm.pipeline.pipeline import Pipeline

import torch
from plants_sm.models.fc.fc import DNN
from plants_sm.models.pytorch_model import PyTorchModel
from torch import nn

from plants_sm.models.lightning_model import InternalLightningModel
from plants_sm.pathway_prediction._fine_tune_ec_number_prediction_model import FineTuneModelECNumber

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

def setup_protbert_models(pipeline_path, device):

    pipeline = Pipeline.load(pipeline_path)

    model = DNN(1024, [2560], 5743, batch_norm=True, last_sigmoid=True)
    model.load_state_dict(torch.load(f"{pipeline_path}/prot_bert.pt"))
    model_1 = PyTorchModel(model=model, loss_function=nn.BCELoss, model_name="ec_number", device=device)

    additional_layers = [1280, 640]

    # learning_rate = 0.0035555738943412697
    # base_layers = [2560]
    batch_size = 64
    input_dim = 1024

    module = FineTuneModelECNumber.load_from_checkpoint(f"{pipeline_path}/enzyme_discrimination.ckpt",
                                            input_dim=input_dim, additional_layers=additional_layers, classification_neurons=1, \
                                                path_to_model=f"{pipeline_path}/prot_bert.pt")
    if device == "cpu":
        accelerator = "cpu"
    else:
        accelerator="gpu"
        device = int(device.replace("cuda:", ""))
        print(device)


    model_2 = InternalLightningModel(module=module,
            batch_size=batch_size,
            devices=device,
            accelerator=accelerator, model_name="enzyme_discrimination")
    
    pipeline.add_models([model_1, model_2])

    return pipeline


def predict_from_csv(dataset_path: str, sequences_field: str,
                    ids_field: str, output_path: str = None, all_data: bool = True,
                    device: str = "cpu", num_gpus: int = 1)-> pd.DataFrame:
    
    """
    Make predictions with a model.

    Parameters
    ----------
    dataset_path: str
        Path to the dataset in a csv format.
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
    results: pandas dataframe
        pandas dataframe with results
    """
    pipeline_path = _download_pipeline_to_cache("ProtBERT pipeline")
    # pipeline_path = "/home/jcapela/plants_ec_number_prediction/PlantsSM/examples/ProtBERT_pipeline"

    dataset = SingleInputDataset.from_csv(dataset_path, representation_field=sequences_field,
                                          instances_ids_field=ids_field)
    
    pipeline = setup_protbert_models(pipeline_path, device)
    
    results_dataframe = _make_predictions_with_model(dataset, pipeline, device, all_data, num_gpus=num_gpus)

    if output_path is not None:
        results_dataframe.to_csv(output_path, index=False)

    return results_dataframe
    

def predict_with_model_from_fasta(fasta_path: str,
                                  output_path: str = None, all_data: bool = True,
                                  device: str = "cpu", num_gpus: int = 1) -> pd.DataFrame:
    """
    Make predictions with a model.

    Parameters
    ----------
    pipeline: str
        Path to the pipeline.
    fasta_path: str
        Path to the fasta file.
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
    results: pandas dataframe
        pandas dataframe with results
    """
    current_directory = os.getcwd()
    temp_csv = os.path.join(current_directory, "temp.csv")
    convert_fasta_to_csv(fasta_path, temp_csv)
    try:
        predictions = predict_from_csv(temp_csv, "sequence", "id", output_path, all_data, device, num_gpus)
        os.remove(temp_csv)
        return predictions
    except Exception as e:
        os.remove(temp_csv)
        raise Exception(e)






