import os
import pickle
import sys
from copy import copy
from unittest import TestCase

import torch
from torch import nn, relu
from torch.optim import Adam

from plants_sm.data_standardization.compounds.deepmol_standardizers import DeepMolStandardizer
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors
from plants_sm.featurization.encoding.one_hot_encoder import OneHotEncoder
from plants_sm.featurization.proteins.bio_embeddings.word2vec import Word2Vec
from plants_sm.models.constants import BINARY
from plants_sm.models.pytorch_model import PyTorchModel
from sklearn.metrics import f1_score, \
    accuracy_score

from plants_sm.tokenisation.compounds.smilespe import AtomLevelTokenizer
from tests import TEST_DIR

environment_name = sys.executable.split('/')[-3]
print('Environment:', environment_name)
os.environ[environment_name] = str(123)
os.environ['PYTHONHASHSEED'] = str(123)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.manual_seed(123)


class InteractionModelDTU(nn.Module):

    def __init__(self, protein_shape, compounds_shape, protein_char_set_n, compound_char_set_n, filters):
        super().__init__()
        self.proteins_embedding = nn.Embedding(num_embeddings=protein_char_set_n + 1, embedding_dim=128, padding_idx=0)
        self.compounds_embedding = nn.Embedding(num_embeddings=compound_char_set_n + 1, embedding_dim=128,
                                                padding_idx=0)
        self.conv1_proteins_1 = nn.Conv1d(protein_shape[1], filters, 4, stride=1,
                                          padding='valid')
        self.conv1_proteins_2 = nn.Conv1d(filters, filters * 2, 6, stride=1, padding='valid')
        self.conv1_proteins_3 = nn.Conv1d(filters * 2, filters * 3, 8, stride=1, padding='valid')
        self.maxpool1_proteins = nn.MaxPool1d(2)

        self.conv1_compounds_1 = nn.Conv1d(compounds_shape[1], filters, 4, stride=1, padding='valid')
        self.conv1_compounds_2 = nn.Conv1d(filters, filters * 2, 6, stride=1, padding='valid')
        self.conv1_compounds_3 = nn.Conv1d(filters * 2, filters * 3, 8,
                                           stride=1, padding='valid')
        self.maxpool1_compounds = nn.MaxPool1d(2)

        self.dense1_interaction = nn.Linear(filters * 3 + filters * 3, 1024)
        self.dense2_interaction = nn.Linear(1024, 1024)
        self.dense3_interaction = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.1)
        self.final_layer = nn.Linear(512, 1)

    def forward(self, x):
        x_proteins = x[0]
        x_proteins = x_proteins.to(torch.int32)
        x_proteins = self.proteins_embedding(x_proteins)
        y = relu(self.conv1_proteins_1(x_proteins))
        y = relu(self.conv1_proteins_2(y))
        y_proteins = relu(self.conv1_proteins_3(y))
        y_proteins = nn.MaxPool1d(y_proteins.shape[2] - 1)(y_proteins)

        x_compounds = x[1]
        x_compounds = x_compounds.to(torch.int32)
        x_compounds = self.compounds_embedding(x_compounds)
        y = relu(self.conv1_compounds_1(x_compounds))
        y = relu(self.conv1_compounds_2(y))
        y_compounds = relu(self.conv1_compounds_3(y))
        y_compounds = nn.MaxPool1d(y_compounds.shape[2] - 1)(y_compounds)

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


class TestDeepDta(TestCase):
    def setUp(self):
        csv_to_read = os.path.join(TEST_DIR, "compound_protein_interaction", "train_small.csv")
        self.dataset_35000_instances_train = MultiInputDataset.from_csv(csv_to_read,
                                                                        representation_fields={"proteins": "SEQ",
                                                                                               "ligands": "SUBSTRATES"},
                                                                        instances_ids_field={"interaction": "index"},
                                                                        labels_field="activity")

        csv_to_read = os.path.join(TEST_DIR, "compound_protein_interaction", "test_small.csv")
        self.dataset_35000_instances_valid = MultiInputDataset.from_csv(csv_to_read,
                                                                        representation_fields={"proteins": "SEQ",
                                                                                               "ligands": "SUBSTRATES"},
                                                                        instances_ids_field={"interaction": "index"},
                                                                        labels_field="activity")

    def test_deep_dta(self):
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

        one_hot = OneHotEncoder(output_shape_dimension=2, alphabet="ARNDCEQGHILKMFPSTWYV").fit(
            self.dataset_35000_instances_train,
            "proteins")
        one_hot.transform(self.dataset_35000_instances_train,
                          "proteins")
        one_hot.transform(self.dataset_35000_instances_valid,
                          "proteins")

        one_hot_compounds = OneHotEncoder(output_shape_dimension=2, tokenizer=AtomLevelTokenizer()).fit(
            self.dataset_35000_instances_train,
            "ligands")
        one_hot_compounds.transform(self.dataset_35000_instances_train, "ligands")

        one_hot_compounds.transform(self.dataset_35000_instances_valid, "ligands")

        input_size_proteins = self.dataset_35000_instances_train.X["proteins"].shape
        input_size_compounds = self.dataset_35000_instances_train.X["ligands"].shape

        n_char_proteins = 20
        n_char_compounds = len(one_hot_compounds.tokens)
        model = InteractionModelDTU(input_size_proteins, input_size_compounds, n_char_proteins, n_char_compounds, 32)

        wrapper = PyTorchModel(model=model, loss_function=nn.BCELoss(), device="cpu",
                               validation_metric=accuracy_score,
                               problem_type=BINARY, batch_size=2, epochs=50,
                               optimizer=Adam(model.parameters(), lr=0.0001), progress=50,
                               logger_path="small_dataset.log")
        wrapper.fit(self.dataset_35000_instances_train, self.dataset_35000_instances_valid)


# @skip("No memory")
class TestConv1D(TestCase):

    def setUp(self) -> None:
        csv_to_read = os.path.join(TEST_DIR, "compound_protein_interaction", "super_train.csv")
        self.dataset_35000_instances_train = MultiInputDataset.from_csv(csv_to_read,
                                                                        representation_fields={"proteins": "SEQ",
                                                                                               "ligands": "SUBSTRATES"},
                                                                        instances_ids_field={"interaction": "index"},
                                                                        labels_field="activity")

        csv_to_read = os.path.join(TEST_DIR, "compound_protein_interaction", "super_test.csv")
        self.dataset_35000_instances_test = MultiInputDataset.from_csv(csv_to_read,
                                                                       representation_fields={"proteins": "SEQ",
                                                                                              "ligands": "SUBSTRATES"},
                                                                       instances_ids_field={"interaction": "index"},
                                                                       labels_field="activity")

        csv_to_read = os.path.join(TEST_DIR, "compound_protein_interaction", "super_valid.csv")
        self.dataset_35000_instances_valid = MultiInputDataset.from_csv(csv_to_read,
                                                                        representation_fields={"proteins": "SEQ",
                                                                                               "ligands": "SUBSTRATES"},
                                                                        instances_ids_field={"interaction": "index"},
                                                                        labels_field="activity")

    def test_dnn_with_pickle(self):
        self.dataset_35000_instances_valid = pickle.load(open("dataset_valid.obj", "rb"))
        self.dataset_35000_instances_train = pickle.load(open("dataset_train.obj", "rb"))

        input_size_proteins = self.dataset_35000_instances_train.X["proteins"].shape[1]
        input_size_compounds = self.dataset_35000_instances_train.X["ligands"].shape[1]
        model = BaselineModel(input_size_proteins, input_size_compounds, [500, 500],
                              [500, 500], [500, 500, 250, 125, 250, 500, 500, 500])

        wrapper = PyTorchModel(model=model, loss_function=nn.BCELoss(),
                               validation_metric=f1_score,
                               problem_type=BINARY, batch_size=50, epochs=2,
                               optimizer=Adam(model.parameters(), lr=0.0001))
        wrapper.fit(self.dataset_35000_instances_train, self.dataset_35000_instances_valid)

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

        one_hot = OneHotEncoder(output_shape_dimension=2, alphabet="ARNDCEQGHILKMFPSTWYV").fit(
            self.dataset_35000_instances_train,
            "proteins")
        one_hot.transform(self.dataset_35000_instances_train,
                          "proteins")

        one_hot_compounds = OneHotEncoder(output_shape_dimension=2, tokenizer=AtomLevelTokenizer()).fit(
            self.dataset_35000_instances_train,
            "ligands")
        one_hot_compounds.transform(self.dataset_35000_instances_train, "ligands")
        one_hot.transform(self.dataset_35000_instances_valid,
                          "proteins")
        one_hot_compounds.transform(self.dataset_35000_instances_valid, "ligands")

        input_size_proteins = self.dataset_35000_instances_train.X["proteins"].shape[1]
        input_size_compounds = self.dataset_35000_instances_train.X["ligands"].shape[1]
        model = BaselineModel(input_size_proteins, input_size_compounds, [500, 500],
                              [500, 500], [500, 500])

        # model = BaselineModel(input_size_proteins, input_size_compounds, [input_size_proteins * 2,
        #                                                                   input_size_proteins * 6,
        #                                                                   input_size_proteins * 8],
        #                       [input_size_compounds * 2,
        #                        input_size_compounds * 6,
        #                        input_size_compounds * 8],
        #                       [input_size_proteins * 2,
        #                        input_size_proteins,
        #                        input_size_proteins // 2, input_size_proteins // 4])

        wrapper = PyTorchModel(model=model, loss_function=nn.BCELoss(),
                               validation_metric=f1_score,
                               problem_type=BINARY, batch_size=50, epochs=50,
                               optimizer=Adam(model.parameters(), lr=0.0001), progress=50,
                               logger_path="small_dataset.log")
        wrapper.fit(self.dataset_35000_instances_train, self.dataset_35000_instances_valid)
        # wrapper.save("test_conv1d.pt")

        # wrapper = PyTorchModel(model=model, loss_function=nn.BCELoss(),
        #                        validation_metric=f1_score,
        #                        problem_type=BINARY, batch_size=50, epochs=2,
        #                        optimizer=Adam(model.parameters(), lr=0.0001), progress=50)
        # wrapper.load(path="test_conv1d.pt")

        # Word2Vec().fit_transform(self.dataset_35000_instances_test,
        #                          "proteins")
        #
        # MAP4Fingerprint(n_jobs=8, dimensions=1024).fit_transform(self.dataset_35000_instances_test, "ligands")

        # one_hot.transform(self.dataset_35000_instances_test,
        #                   "proteins")
        # one_hot_compounds.transform(self.dataset_35000_instances_test, "ligands")
        # predictions = wrapper.predict(self.dataset_35000_instances_test)
        #
        # ModelReport(wrapper, BINARY, self.dataset_35000_instances_test).generate_metrics_report()

    def test_pickle_dataset(self):
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

        Word2Vec().fit_transform(self.dataset_35000_instances_train,
                                 "proteins")

        DeepMolDescriptors(n_jobs=8).fit_transform(self.dataset_35000_instances_train, "ligands")

        Word2Vec().fit_transform(self.dataset_35000_instances_valid,
                                 "proteins")

        DeepMolDescriptors(n_jobs=8).fit_transform(self.dataset_35000_instances_valid, "ligands")
        file_pi = open('dataset_train.obj', 'wb')
        pickle.dump(self.dataset_35000_instances_train, file_pi)
        file_pi = open('dataset_valid.obj', 'wb')
        pickle.dump(self.dataset_35000_instances_valid, file_pi)

    def test_get_dataset(self):
        file_pi = open('dataset.obj', 'rb')
        dataset = pickle.load(file_pi)
        print(dataset.X)
