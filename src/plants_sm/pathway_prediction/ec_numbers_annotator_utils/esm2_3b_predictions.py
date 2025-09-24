import os
import pandas as pd

from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
from plants_sm.pathway_prediction.ec_numbers_annotator_utils._models_predictions_utils import _make_predictions_with_model
from plants_sm.pathway_prediction.ec_numbers_annotator_utils._utils import _download_pipeline_to_cache, convert_fasta_to_csv
from plants_sm.pipeline.pipeline import Pipeline

import torch
from plants_sm.models.fc.fc import DNN
from plants_sm.models.pytorch_model import PyTorchModel
from torch import nn

from plants_sm.models.lightning_model import InternalLightningModel
from plants_sm.pathway_prediction._fine_tune_ec_number_prediction_model import ModelECNumber

def setup_esm2_models(pipeline_path, device, num_gpus=1):

    pipeline = Pipeline.load(pipeline_path)

    pipeline.steps["place_holder"][-1].device = device
    
    if "cuda" in device:
        if device == "cuda":
            pipeline.steps["place_holder"][-1].num_gpus = num_gpus
        else:
            pipeline.steps["place_holder"][-1].num_gpus = 1

        pipeline.steps["place_holder"][-1].is_ddf = True
    
    model = DNN(2560, [2560], 5743, batch_norm=True, last_sigmoid=True)
    model.load_state_dict(torch.load(f"{pipeline_path}/esm2_3b.pt"))
    model_1 = PyTorchModel(model=model, loss_function=nn.BCELoss, model_name="ec_number", device=device)

    additional_layers = [2560, 1280, 1280]

    base_layers = [2560]
    batch_size = 64
    input_dim = 2560

    module = ModelECNumber.load_from_checkpoint(f"{pipeline_path}/enzyme_discrimination.ckpt",
                                            input_dim=input_dim, layers=base_layers + additional_layers, classification_neurons=1, \
                                            scheduler=False
                                            )
    if device == "cpu":
        accelerator = "cpu"
    else:
        accelerator="gpu"
        device = int(device.replace("cuda:", ""))


    model_2 = InternalLightningModel(module=module,
            batch_size=batch_size,
            devices=device,
            accelerator=accelerator, model_name="enzyme_discrimination")
    
    pipeline.add_models([model_1, model_2])

    return pipeline

def predict_with_esm2_from_csv(dataset_path: str, sequences_field: str,
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
    pipeline_path = _download_pipeline_to_cache("ESM2 pipeline")
    # pipeline_path = "/home/jcapela/plants_ec_number_prediction/PlantsSM/examples/ESM1b_pipeline"

    dataset = SingleInputDataset.from_csv(dataset_path, representation_field=sequences_field,
                                          instances_ids_field=ids_field)
    
    pipeline = setup_esm2_models(pipeline_path, device)
    
    results_dataframe = _make_predictions_with_model(dataset, pipeline, device, all_data, num_gpus=num_gpus)

    if output_path is not None:
        results_dataframe.to_csv(output_path, index=False)

    return results_dataframe

def predict_with_esm2_from_dataframe(dataset: pd.DataFrame, sequences_field: str,
                    ids_field: str, output_path: str = None, all_data: bool = True,
                    device: str = "cpu", num_gpus: int = 1)-> pd.DataFrame:
    
    """
    Make predictions with a model.

    Parameters
    ----------
    dataset: pd.DataFrame
        pd.DataFrame of the dataset
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
    pipeline_path = _download_pipeline_to_cache("ESM2 pipeline")

    dataset = SingleInputDataset(dataset, representation_field=sequences_field,
                                          instances_ids_field=ids_field)
    
    pipeline = setup_esm2_models(pipeline_path, device, num_gpus=num_gpus)
    
    results_dataframe = _make_predictions_with_model(dataset, pipeline, device, all_data, num_gpus=num_gpus)

    if output_path is not None:
        results_dataframe.to_csv(output_path, index=False)

    return results_dataframe
    

def predict_with_esm2_from_fasta(fasta_path: str,
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
        predictions = predict_with_esm2_from_csv(temp_csv, "sequence", "id", output_path, all_data, device, num_gpus)
        os.remove(temp_csv)
        return predictions
    except Exception as e:
        os.remove(temp_csv)
        raise Exception(e)