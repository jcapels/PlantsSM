
from unittest import TestCase
from tests import TEST_DIR
import pandas as pd
import os
from plants_sm.ml.data_structures.dataset.multi_input_dataset import MultiInputDataset

from plants_sm.ml.models.lightning_model import InternalLightningModel
from ._multi_modal_model import MultiModalModel
from lightning.pytorch.callbacks import EarlyStopping
from ._interaction_model import InteractionModel

class TestMultiModalLightningModel(TestCase):

    def setUp(self) -> None:
        
        multi_input_dataset_csv = os.path.join(TEST_DIR, "data/train_dataset_multi_modal.csv")

        self.train_dataset = MultiInputDataset.from_csv(file_path=multi_input_dataset_csv,
                                                        representation_field={"proteins": "sequence",
                                                                            "ligands": "SMILES",
                                                                            "reactions": "reaction_SMILES"},
                                                        instances_ids_field={"proteins": "uniprot_id",
                                                                            "ligands": "CHEBI_ID",
                                                                            "reactions": "RHEA_ID"},
                                                        labels_field="interaction")

        multi_input_dataset_csv = os.path.join(TEST_DIR, "data/val_dataset_multi_modal.csv")

        self.validation_dataset = MultiInputDataset.from_csv(file_path=multi_input_dataset_csv,
                                                        representation_field={"proteins": "sequence",
                                                                            "ligands": "SMILES",
                                                                            "reactions": "reaction_SMILES"},
                                                        instances_ids_field={"proteins": "uniprot_id",
                                                                            "ligands": "CHEBI_ID",
                                                                            "reactions": "RHEA_ID"},
                                                        labels_field="interaction")
        
        multi_input_dataset_csv = os.path.join(TEST_DIR, "data/test_dataset_multi_modal.csv")

        self.test_dataset = MultiInputDataset.from_csv(file_path=multi_input_dataset_csv,
                                                        representation_field={"proteins": "sequence",
                                                                            "ligands": "SMILES",
                                                                            "reactions": "reaction_SMILES"},
                                                        instances_ids_field={"proteins": "uniprot_id",
                                                                            "ligands": "CHEBI_ID",
                                                                            "reactions": "RHEA_ID"},
                                                        labels_field="interaction")
    
    def test_multi_modal_lightning_model(self):
        self.train_dataset.load_features(os.path.join(TEST_DIR, "data/not_to_push_data/train_dataset_multi_modal_features"))
        self.validation_dataset.load_features(os.path.join(TEST_DIR, "data/not_to_push_data/validation_dataset_multi_modal_features"))
        self.test_dataset.load_features(os.path.join(TEST_DIR, "data/not_to_push_data/test_dataset_multi_modal_features"))

        module = MultiModalModel(protein_model_path=os.path.join(TEST_DIR, "data", "not_to_push_data", "esm1b.pt"), 
                                 compounds_model_path=os.path.join(TEST_DIR, "data", "not_to_push_data", "np_classifier.ckpt"), 
                                 reactions_model_path=os.path.join(TEST_DIR, "data", "not_to_push_data", "smiles_reaction_bert_ec_model.pt"), 
                                 transfer_learning=True)

        # Create the early stopping callback
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',  # metric to monitor
            patience=5,          # number of epochs with no improvement after which training will be stopped
            verbose=True,
            mode='min'           # stop when the metric is minimized
        )

        model = InternalLightningModel(module=module, max_epochs=1,
                batch_size=3,
                devices=[0],
                accelerator="gpu",
                callbacks=[early_stopping_callback]
                )
        
        model.fit(self.train_dataset, validation_dataset=self.validation_dataset)
        predictions = model.predict(self.test_dataset)

    def test_interaction_model(self):

        multi_input_dataset_csv = os.path.join(TEST_DIR, "data/train_dataset_multi_modal.csv")

        self.train_dataset = MultiInputDataset.from_csv(file_path=multi_input_dataset_csv,
                                                        representation_field={"proteins": "sequence",
                                                                            "ligands": "SMILES"},
                                                        instances_ids_field={"proteins": "uniprot_id",
                                                                            "ligands": "CHEBI_ID"},
                                                        labels_field="interaction")

        multi_input_dataset_csv = os.path.join(TEST_DIR, "data/val_dataset_multi_modal.csv")

        self.validation_dataset = MultiInputDataset.from_csv(file_path=multi_input_dataset_csv,
                                                        representation_field={"proteins": "sequence",
                                                                            "ligands": "SMILES"},
                                                        instances_ids_field={"proteins": "uniprot_id",
                                                                            "ligands": "CHEBI_ID"},
                                                        labels_field="interaction")
        
        multi_input_dataset_csv = os.path.join(TEST_DIR, "data/test_dataset_multi_modal.csv")

        self.test_dataset = MultiInputDataset.from_csv(file_path=multi_input_dataset_csv,
                                                        representation_field={"proteins": "sequence",
                                                                            "ligands": "SMILES"},
                                                        instances_ids_field={"proteins": "uniprot_id",
                                                                            "ligands": "CHEBI_ID"},
                                                        labels_field="interaction")
        
        self.train_dataset.load_features(os.path.join(TEST_DIR, "data/not_to_push_data/train_dataset_multi_modal_features"), "ligands")
        self.validation_dataset.load_features(os.path.join(TEST_DIR, "data/not_to_push_data/validation_dataset_multi_modal_features"), "ligands")
        self.test_dataset.load_features(os.path.join(TEST_DIR, "data/not_to_push_data/test_dataset_multi_modal_features"), "ligands")

        self.train_dataset.load_features(os.path.join(TEST_DIR, "data/not_to_push_data/train_dataset_multi_modal_features"), "proteins")
        self.validation_dataset.load_features(os.path.join(TEST_DIR, "data/not_to_push_data/validation_dataset_multi_modal_features"), "proteins")
        self.test_dataset.load_features(os.path.join(TEST_DIR, "data/not_to_push_data/test_dataset_multi_modal_features"), "proteins")

        module = InteractionModel(protein_model_path=os.path.join(TEST_DIR, "data", "not_to_push_data", "esm1b.pt"), compounds_model_path=os.path.join(TEST_DIR, "data", "not_to_push_data", "np_classifier.ckpt"), transfer_learning=True)

        # Create the early stopping callback
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',  # metric to monitor
            patience=5,          # number of epochs with no improvement after which training will be stopped
            verbose=True,
            mode='min'           # stop when the metric is minimized
        )

        model = InternalLightningModel(module=module, max_epochs=1,
                batch_size=128,
                devices=[0],
                accelerator="gpu",
                callbacks=[early_stopping_callback]
                )
        predictions = model.predict(self.test_dataset)

        





    
