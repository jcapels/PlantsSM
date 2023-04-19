from logging import Logger
import os
import numpy as np
import torch
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
from plants_sm.models.cnn.cnn import CNN1D
from plants_sm.models.fc.fc import DenseNet
from plants_sm.models.pytorch_model import PyTorchModel
from torch import nn

import pandas as pd
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score


logger = Logger("train_model")

def generate_metrics(dataset, model):

    proba = model.predict(dataset)

    hamming_loss_ = hamming_loss(dataset.y, proba)
    accuracy_score_ = accuracy_score(dataset.y, proba)
    precision_score_ = precision_score(dataset.y, proba, average="samples")
    recall_score_ = recall_score(dataset.y, proba, average="samples")
    f1_score_ = f1_score(dataset.y, proba, average="samples")

    return {"hamming_loss": hamming_loss_, "accuracy_score": accuracy_score_,
            "precision_score": precision_score_, "recall_score": recall_score_, "f1_score": f1_score_}


def prepare_dataset(train_dataset_path, validation_dataset_path, test_dataset_path):
    
    logger.info("Loading the training and validation datasets")
    train_dataset = SingleInputDataset.from_csv(train_dataset_path,
                                                instances_ids_field="accession", representation_field="sequence",
                                                labels_field=slice(8, 2779), features_fields=slice(2780, -1))

    validation_dataset = SingleInputDataset.from_csv(validation_dataset_path,
                                                     instances_ids_field="accession", representation_field="sequence",
                                                     labels_field=slice(8, 2779), features_fields=slice(2780, -1))
    
    test_dataset = SingleInputDataset.from_csv(test_dataset_path,
                                                instances_ids_field="accession", representation_field="sequence",
                                                labels_field=slice(8, 2779), features_fields=slice(2780, -1))

    logger.info("datasets loaded")
    return train_dataset, validation_dataset, test_dataset

# Train the model
def get_ratio(y):
    ratio = np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        ratio[i] = np.sum(y[:, i] == 0) / np.sum(y[:, i] == 1)
    return ratio

def train_model(train_dataset_path, validation_dataset_path, 
                test_dataset_path, model_path, base_path, model, descriptor):

    train_dataset, validation_dataset, test_dataset = prepare_dataset(train_dataset_path, validation_dataset_path, test_dataset_path)

    # ratio = get_ratio(train_dataset.y)

    # Train the model
    logger.info("Training the model")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # pos_weight = torch.tensor(ratio).to("cuda:0")

    model = PyTorchModel(batch_size=16, epochs=50,
                         loss_function=nn.BCEWithLogitsLoss(), optimizer=optimizer,
                         model=model,
                         device="cuda:0", logger_path="./logs.log",
                         progress=200, model_name=f"{model.__class__.__name__}_{descriptor}",
                         patience=2)
    
    model.fit(train_dataset=train_dataset, validation_dataset=validation_dataset)
    model.model.last_sigmoid = True
    model.save(model_path)

    # Evaluate the model
    logger.info("Evaluating the model")

    train_metrics = generate_metrics(train_dataset, model)
    validation_metrics = generate_metrics(validation_dataset, model)
    metrics_test = generate_metrics(test_dataset, model)

    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Validation metrics: {validation_metrics}")
    logger.info(f"Test metrics: {metrics_test}")

    os.makedirs(os.path.join(base_path, "metrics"), exist_ok=True)

    #create dataframe with the metrics for all datasets and models

    if os.path.exists(os.path.join(base_path, "metrics", "metrics.csv")):
        metrics = pd.read_csv(os.path.join(base_path, "metrics", "metrics.csv"))
    else:
        metrics = pd.DataFrame(columns=["model", "metric", "train", "validation", "test"])

    dataset_metrics = [train_metrics, validation_metrics, metrics_test]
    dataset_names = ["train", "validation", "test"]
    for metric in train_metrics.keys():
        row = len(metrics)
        metrics.at[row, "model"] = f"{model.model.__class__.__name__}_{descriptor}"
        metrics.at[row, "metric"] = metric
        for i, dataset_metric in enumerate(dataset_metrics):
            metrics.at[row, dataset_names[i]] = dataset_metric[metric]

    metrics.to_csv(os.path.join(base_path, "metrics", "metrics.csv"), index=False)

    model.history["loss"].to_csv(os.path.join(model_path, f"{model.model.__class__.__name__}_loss.csv"), index=False)


def run_for_esm2(folder):
    base_directory = "/scratch/jribeiro/ec_number_prediction/esm2_data/"
    cnn_model = CNN1D([5120, 3840, 2880], [160, 80], [2, 2], 2771, False)
    fc_model = DenseNet(320, [640, 1280, 2560, 5120], 2771, False)

    train_dataset_path = os.path.join(base_directory, folder, f"train_{folder}_UR50D.csv")
    validation_dataset_path = os.path.join(base_directory, folder, f"validation_{folder}_UR50D.csv")
    test_dataset_path = os.path.join(base_directory, folder, f"test_{folder}_UR50D.csv")

    for model in [cnn_model, fc_model]:
        os.makedirs(os.path.join(base_directory, folder, f"{model.__class__.__name__}"), exist_ok=True)
        model_path = os.path.join(base_directory, folder, f"{model.__class__.__name__}", "model")
        train_model(train_dataset_path, validation_dataset_path, test_dataset_path, model_path, base_directory, model, folder)

def run_test():
    train_dataset_path = os.path.join("train_test.csv")
    validation_dataset_path = os.path.join("validation_test.csv")
    test_dataset_path = os.path.join(f"test_test.csv")

    model = CNN1D([5120, 3840, 2880], [160, 80], [2, 2], 2771, False)

    os.makedirs(os.path.join("CNN1D"), exist_ok=True)
    model_path = os.path.join("CNN1D", "model")

    train_model(train_dataset_path, validation_dataset_path, test_dataset_path, model_path, "./", model, "esm2_8M")

if __name__ == "__main__":
    run_test()
    # run_for_esm2("esm2_t30_150M")