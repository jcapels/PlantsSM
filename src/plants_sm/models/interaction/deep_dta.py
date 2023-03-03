import torch
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from torch import nn, relu
from transformers import Conv1D
from tensorflow.keras.layers import Conv1D, concatenate, Embedding, GlobalMaxPooling1D, Dropout, Dense
import tensorflow as tf


class BalAccScore(tf.keras.callbacks.Callback):

    def __init__(self, X, y):
        super(BalAccScore, self).__init__()
        self.X = X
        self.y = y

    def on_train_begin(self, logs={}):
        self.balanced_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        y_predict = tf.argmax(self.model.predict(self.X), axis=1)
        y_true = tf.argmax(self.y, axis=1)
        balacc = balanced_accuracy_score(y_true.numpy(), y_predict.numpy())
        self.balanced_accuracy.append(round(balacc, 6))
        logs["val_bal_acc"] = balacc
        keys = list(logs.keys())

        print("\n ------ validation balanced accuracy score: %f ------\n" % balacc)


def balanced_accuracy(y_true, y_pred):
    from tensorflow.keras import backend as K
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_keras = true_positives / (possible_positives + K.epsilon())
        return recall_keras

    def specificity(y_true, y_pred):
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        return tn / (tn + fp + K.epsilon())

    balanced_accuracy = (recall(y_true, y_pred) + specificity(y_true, y_pred)) / 2

    return balanced_accuracy


class DeepDTATensorflow:

    def __init__(self, PROT_LEN, SMILE_LEN, CHAR_PROT_SET_SIZE, CHAR_SMI_SET_SIZE, NUM_FILTERS):

        self.PROT_LEN = PROT_LEN
        self.SMILE_LEN = SMILE_LEN
        self.CHAR_SMI_SET_SIZE = CHAR_SMI_SET_SIZE
        self.CHAR_PROT_SET_SIZE = CHAR_PROT_SET_SIZE
        self.NUM_FILTERS = NUM_FILTERS
        self.model = self.build_model()

    def build_model(self):
        X_PROT = Input(shape=(self.PROT_LEN,), dtype='int32')
        X_MET = Input(shape=(self.SMILE_LEN,), dtype='int32')

        encode_protein = Embedding(input_dim=self.CHAR_PROT_SET_SIZE, output_dim=128, input_length=self.PROT_LEN)(
            X_PROT)
        encode_protein = Conv1D(filters=self.NUM_FILTERS, kernel_size=4, activation='relu', padding='valid', strides=1)(
            encode_protein)
        encode_protein = Conv1D(filters=self.NUM_FILTERS * 2, kernel_size=8, activation='relu', padding='valid',
                                strides=1)(
            encode_protein)
        encode_protein = Conv1D(filters=self.NUM_FILTERS * 3, kernel_size=10, activation='relu', padding='valid',
                                strides=1)(encode_protein)
        encode_protein = GlobalMaxPooling1D()(encode_protein)

        encode_smiles = Embedding(input_dim=self.CHAR_SMI_SET_SIZE, output_dim=128, input_length=self.SMILE_LEN)(X_MET)
        encode_smiles = Conv1D(filters=self.NUM_FILTERS, kernel_size=4, activation='relu', padding='valid', strides=1)(
            encode_smiles)
        encode_smiles = Conv1D(filters=self.NUM_FILTERS * 2, kernel_size=6, activation='relu', padding='valid',
                               strides=1)(
            encode_smiles)
        encode_smiles = Conv1D(filters=self.NUM_FILTERS * 3, kernel_size=8, activation='relu', padding='valid',
                               strides=1)(
            encode_smiles)
        encode_smiles = GlobalMaxPooling1D()(encode_smiles)

        encode_interaction = concatenate([encode_smiles, encode_protein],
                                         axis=-1)

        # Fully connected
        FC1 = Dense(1024, activation='relu')(encode_interaction)
        FC2 = Dropout(0.1)(FC1)
        FC2 = Dense(1024, activation='relu')(FC2)
        FC2 = Dropout(0.1)(FC2)
        FC2 = Dense(512, activation='relu')(FC2)

        # And add a logistic regression on top
        predictions = Dense(1, activation='sigmoid')(FC2)  # kernel_initializer='normal"

        interactionModel = Model(inputs=[X_PROT, X_MET], outputs=[predictions])

        interactionModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=[balanced_accuracy])
        return interactionModel


class DeepDTA(nn.Module):

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
