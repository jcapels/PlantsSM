import os
from copy import copy
from unittest import TestCase, skip

import torch
from torch import nn, relu
from torch.optim import Adam

from plants_sm.data_standardization.compounds.deepmol_standardizers import DeepMolStandardizer
from plants_sm.data_standardization.compounds.padding import SMILESPadder
from plants_sm.data_standardization.proteins.padding import SequencePadder
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.featurization.compounds.map4_fingerprint import MAP4Fingerprint
from plants_sm.featurization.one_hot_encoder import OneHotEncoder
from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert
from plants_sm.models.constants import BINARY
from plants_sm.models.pytorch_model import PyTorchModel
from sklearn.metrics import balanced_accuracy_score

from tests import TEST_DIR


class InteractionModelDTU(nn.Module):

    def __init__(self, protein_shape, compounds_shape):
        super().__init__()
        self.conv1_proteins_1 = nn.Conv1d(protein_shape[1], protein_shape[1] * 2, 4, stride=1,
                                          padding='valid')
        self.conv1_proteins_2 = nn.Conv1d(protein_shape[1] * 2, protein_shape[1] * 3, 6, stride=1, padding='valid')
        self.conv1_proteins_3 = nn.Conv1d(protein_shape[1] * 3, protein_shape[1] * 4, 8, stride=1, padding='valid')
        self.maxpool1_proteins = nn.MaxPool1d(11)

        self.conv1_compounds_1 = nn.Conv1d(compounds_shape[1], compounds_shape[1] * 2, 4, stride=1, padding='valid')
        self.conv1_compounds_2 = nn.Conv1d(compounds_shape[1] * 2, compounds_shape[1] * 3, 6, stride=1, padding='valid')
        self.conv1_compounds_3 = nn.Conv1d(compounds_shape[1] * 3, compounds_shape[1] * 4, 8,
                                           stride=1, padding='valid')
        self.maxpool1_compounds = nn.MaxPool1d(11)

        self.dense1_interaction = nn.Linear(compounds_shape[1] * 4 + protein_shape[1] * 4, 1024)
        self.dense2_interaction = nn.Linear(1024, 1024)
        self.dense3_interaction = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.1)
        self.final_layer = nn.Linear(512, 1)

    def forward(self, x):
        x_proteins = x[0]
        x_proteins = x_proteins.unsqueeze(2)
        y = relu(self.conv1_proteins_1(x_proteins))
        y = relu(self.conv1_proteins_2(y))
        y = relu(self.conv1_proteins_3(y))
        y_proteins = self.maxpool1_proteins(y)

        x_compounds = x[1]
        x_compounds = x_compounds.unsqueeze(2)
        y = relu(self.conv1_compounds_1(x_compounds))
        y = relu(self.conv1_compounds_2(y))
        y = relu(self.conv1_compounds_3(y))
        y_compounds = self.maxpool1_compounds(y)

        y = torch.cat([y_proteins, y_compounds], dim=1)
        y = y.reshape(y.shape[0], y.shape[1])
        y = relu(self.dense1_interaction(y))
        y = self.dropout(y)
        y = relu(self.dense2_interaction(y))
        y = self.dropout(y)
        y = relu(self.dense3_interaction(y))
        y = self.dropout(y)
        y = torch.sigmoid(self.final_layer(y))
        return y


class BaselineModel(nn.Module):
    def __init__(self, input_size_proteins, input_size_compounds, hidden_layers_proteins,
                 hidden_layers_compounds, hidden_layers_interaction):
        super().__init__()
        self.hidden_layers_proteins = hidden_layers_proteins
        self.hidden_layers_compounds = hidden_layers_compounds
        self.hidden_layers_interaction = hidden_layers_interaction
        layers_neurons = input_size_proteins * 2
        linear_input = copy(layers_neurons)
        self.dense_proteins1 = nn.Linear(input_size_proteins, layers_neurons)
        for i, hidden_size in enumerate(hidden_layers_proteins):
            self.add_module(
                'fc_proteins{}'.format(i),
                nn.Linear(linear_input, hidden_size)
            )
            linear_input = copy(hidden_size)

        proteins_final_layer = copy(linear_input)

        layers_neurons = input_size_compounds * 2
        self.dense_compounds1 = nn.Linear(input_size_compounds, layers_neurons)
        linear_input = copy(layers_neurons)
        for i, hidden_size in enumerate(hidden_layers_compounds):
            self.add_module(
                'fc_compounds{}'.format(i),
                nn.Linear(linear_input, hidden_size)
            )
            linear_input = copy(hidden_size)
        compounds_final_layer = copy(linear_input)

        self.dense_interaction_layer1 = nn.Linear(proteins_final_layer + compounds_final_layer, layers_neurons)
        linear_input = copy(layers_neurons)
        for i, hidden_size in enumerate(hidden_layers_interaction):
            self.add_module(
                'fc_interaction{}'.format(i),
                nn.Linear(linear_input, hidden_size)
            )
            linear_input = copy(hidden_size)

        self.final_layer = nn.Linear(linear_input, 1)

    def forward(self, x):
        x_proteins = x[0]
        x_proteins = relu(self.dense_proteins1(x_proteins))
        for i, layer in enumerate(self.hidden_layers_proteins):
            x_proteins = relu(getattr(self, 'fc_proteins{}'.format(i))(x_proteins))

        x_compounds = x[1]
        x_compounds = relu(self.dense_compounds1(x_compounds))
        for i, layer in enumerate(self.hidden_layers_compounds):
            x_compounds = relu(getattr(self, 'fc_compounds{}'.format(i))(x_compounds))

        x_interaction = torch.cat([x_proteins, x_compounds], dim=1)
        x_interaction = relu(self.dense_interaction_layer1(x_interaction))
        for i, layer in enumerate(self.hidden_layers_interaction):
            x_interaction = relu(getattr(self, 'fc_interaction{}'.format(i))(x_interaction))
        y = torch.sigmoid(self.final_layer(x_interaction))
        return y


# @skip("No memory")
class TestConv1D(TestCase):

    def setUp(self) -> None:
        csv_to_read = os.path.join(TEST_DIR, "performance_datasets", "phosphatase_chiral_binary_train.csv")
        self.dataset_35000_instances_train = MultiInputDataset.from_csv(csv_to_read,
                                                                        representation_fields={"proteins": "SEQ",
                                                                                               "ligands": "SUBSTRATES"},
                                                                        instances_ids_field={"interaction": "ids"},
                                                                        labels_field="Conversion")

        csv_to_read = os.path.join(TEST_DIR, "performance_datasets", "phosphatase_chiral_binary_test.csv")
        self.dataset_35000_instances_test = MultiInputDataset.from_csv(csv_to_read,
                                                                       representation_fields={"proteins": "SEQ",
                                                                                              "ligands": "SUBSTRATES"},
                                                                       instances_ids_field={"interaction": "ids"},
                                                                       labels_field="Conversion")

        csv_to_read = os.path.join(TEST_DIR, "performance_datasets", "phosphatase_chiral_binary_val.csv")
        self.dataset_35000_instances_valid = MultiInputDataset.from_csv(csv_to_read,
                                                                        representation_fields={"proteins": "SEQ",
                                                                                               "ligands": "SUBSTRATES"},
                                                                        instances_ids_field={"interaction": "ids"},
                                                                        labels_field="Conversion")

    def test_conv1d(self):
        HEAVY_STANDARDIZATION = {
            'remove_isotope'.upper(): True,
            'NEUTRALISE_CHARGE'.upper(): True,
            'remove_stereo'.upper(): True,
            'keep_biggest'.upper(): True,
            'add_hydrogen'.upper(): True,
            'kekulize'.upper(): False,
            'neutralise_charge_late'.upper(): True
        }

        kwargs = {"params": HEAVY_STANDARDIZATION}

        DeepMolStandardizer(preset="custom_standardizer", kwargs=kwargs, n_jobs=8).fit_transform(
            self.dataset_35000_instances_train,
            "ligands")

        ProteinStandardizer(n_jobs=8).fit_transform(self.dataset_35000_instances_valid, "proteins")

        DeepMolStandardizer(preset="custom_standardizer", kwargs=kwargs, n_jobs=8).fit_transform(
            self.dataset_35000_instances_valid,
            "ligands")

        ProteinStandardizer(n_jobs=8).fit_transform(self.dataset_35000_instances_train, "proteins")

        ProtBert(device="cuda").fit_transform(self.dataset_35000_instances_train,
                                              "proteins")
        MAP4Fingerprint(n_jobs=8, dimensions=1024).fit_transform(self.dataset_35000_instances_train, "ligands")

        ProtBert(device="cuda").fit_transform(self.dataset_35000_instances_valid,
                                              "proteins")
        MAP4Fingerprint(n_jobs=8, dimensions=1024).fit_transform(self.dataset_35000_instances_valid, "ligands")

        input_size_proteins = self.dataset_35000_instances_train.X["proteins"].shape[1]
        input_size_compounds = self.dataset_35000_instances_train.X["ligands"].shape[1]
        model = BaselineModel(input_size_proteins, input_size_compounds, [input_size_proteins * 3,
                                                                          input_size_proteins * 4],
                              [input_size_compounds * 3,
                               input_size_compounds * 4],
                              [input_size_compounds * 2,
                               input_size_compounds,
                               input_size_compounds // 2])
        wrapper = PyTorchModel(model=model, loss_function=nn.BCELoss(),
                               validation_metric=balanced_accuracy_score,
                               problem_type=BINARY, batch_size=75, epochs=50,
                               optimizer=Adam(model.parameters(), lr=0.0001))
        wrapper.fit(self.dataset_35000_instances_train, self.dataset_35000_instances_valid)

    def test_padding(self):
        HEAVY_STANDARDIZATION = {
            'remove_isotope'.upper(): True,
            'NEUTRALISE_CHARGE'.upper(): True,
            'remove_stereo'.upper(): True,
            'keep_biggest'.upper(): True,
            'add_hydrogen'.upper(): True,
            'kekulize'.upper(): False,
            'neutralise_charge_late'.upper(): True
        }

        kwargs = {"params": HEAVY_STANDARDIZATION}

        DeepMolStandardizer(preset="custom_standardizer", kwargs=kwargs, n_jobs=8).fit_transform(
            self.dataset_35000_instances_train,
            "ligands")
        ProteinStandardizer(n_jobs=8).fit_transform(self.dataset_35000_instances_train, "proteins")

        smiles_padder = SMILESPadder(n_jobs=8).fit(self.dataset_35000_instances_train, "ligands")
        smiles_padder.transform(self.dataset_35000_instances_train, "ligands")

        sequence_padder = SequencePadder(n_jobs=8).fit(self.dataset_35000_instances_train, "proteins")
        sequence_padder.transform(self.dataset_35000_instances_train, "proteins")

        proteins_one_hot = OneHotEncoder(n_jobs=8).fit(self.dataset_35000_instances_train, "proteins")
        proteins_one_hot.transform(self.dataset_35000_instances_train, "proteins")
        compounds_one_hot = OneHotEncoder(n_jobs=8).fit(self.dataset_35000_instances_train, "ligands")
        compounds_one_hot.transform(self.dataset_35000_instances_train, "ligands")

        DeepMolStandardizer(preset="custom_standardizer", kwargs=kwargs, n_jobs=8).fit_transform(
            self.dataset_35000_instances_valid,
            "ligands")
        ProteinStandardizer(n_jobs=8).fit_transform(self.dataset_35000_instances_valid, "proteins")

        smiles_padder.transform(self.dataset_35000_instances_valid, "ligands")
        sequence_padder.transform(self.dataset_35000_instances_valid, "proteins")
        proteins_one_hot.transform(self.dataset_35000_instances_valid, "proteins")
        compounds_one_hot.transform(self.dataset_35000_instances_valid, "ligands")

        proteins_shape = self.dataset_35000_instances_train.X["proteins"].shape
        compounds_shape = self.dataset_35000_instances_train.X["ligands"].shape
        dta = InteractionModelDTU(proteins_shape, compounds_shape)
        wrapper = PyTorchModel(model=dta, loss_function=nn.BCELoss(),
                               validation_metric=balanced_accuracy_score,
                               problem_type=BINARY, batch_size=75, epochs=50,
                               optimizer=Adam(dta.parameters(), lr=0.0001))
        wrapper.fit(self.dataset_35000_instances_train, self.dataset_35000_instances_valid)
