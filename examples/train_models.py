from logging import Logger
import os
import numpy as np
import torch
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
from plants_sm.models.cnn.cnn import CNN1D
from plants_sm.models.pytorch_model import PyTorchModel
from torch import nn


def train_model(train_dataset_path, validation_dataset_path, model_path):
    logger = Logger("train_model")
    # Load the training and validation datasets

    logger.info("Loading the training and validation datasets")
    train_dataset = SingleInputDataset.from_csv(train_dataset_path,
                                                instances_ids_field="accession", representation_field="sequence",
                                                labels_field=slice(8, 2779), features_fields=slice(2780, -1))

    validation_dataset = SingleInputDataset.from_csv(validation_dataset_path,
                                                     instances_ids_field="accession", representation_field="sequence",
                                                     labels_field=slice(8, 2779), features_fields=slice(2780, -1))

    logger.info("datasets loaded")

    # Train the model
    def get_ratio(y):
        ratio = np.zeros(y.shape[1])
        for i in range(y.shape[1]):
            ratio[i] = np.sum(y[:, i] == 0) / np.sum(y[:, i] == 1)
        return ratio

    ratio = get_ratio(train_dataset.y)

    # Train the model
    logger.info("Training the model")

    cnn_model = CNN1D([5120, 3000], [160], [2], 2771, False)

    optimizer = torch.optim.Adam(params=cnn_model.parameters(), lr=0.001)

    pos_weight = torch.tensor(ratio).to("cuda:0")

    model = PyTorchModel(batch_size=200, epochs=1,
                         loss_function=nn.BCEWithLogitsLoss(pos_weight=pos_weight), optimizer=optimizer,
                         model=cnn_model,
                         device="cuda:0", logger_path="./logs.log",
                         progress=200)
    model.fit(train_dataset=train_dataset, validation_dataset=validation_dataset)
    model.save(model_path)


if __name__ == "__main__":
    esm2_data_folders = os.listdir("esm2_data")

    for folder in esm2_data_folders:
        train_dataset_path = os.path.join("esm2_data", folder, f"train_{folder}_URD50.csv")
        validation_dataset_path = os.path.join("esm2_data", folder, f"validation_{folder}_URD50.csv")
        # model_path = os.path.join("esm2_data", folder, "model")
        os.makedirs(os.path.join("esm2_data", folder, "CNN1D"), exist_ok=True)
        model_path = os.path.join("esm2_data", folder, "CNN1D", "model")
        train_model(train_dataset_path, validation_dataset_path, model_path)
    

