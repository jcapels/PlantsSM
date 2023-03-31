import copy
import os
from unittest import TestCase

from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam

from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_structures.dataset import SingleInputDataset
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.featurization.proteins.bio_embeddings.word2vec import Word2Vec
from plants_sm.models.constants import BINARY
from plants_sm.models.pytorch_model import PyTorchModel
from plants_sm.models.tensorflow_model import TensorflowModel
from tests import TEST_DIR
from unit_tests.models._utils import TestPytorchBaselineModel, ToyTensorflowModel


class TestDatasetModel(TestCase):

    def setUp(self) -> None:
        self.single_input_dataset_csv = os.path.join(TEST_DIR, "data", "proteins.csv")

        self.single_input_dataset = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                                                representation_field="sequence",
                                                                instances_ids_field="id",
                                                                labels_field="y",
                                                                batch_size=1)

        self.single_input_dataset_val = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                                                    representation_field="sequence",
                                                                    instances_ids_field="id",
                                                                    labels_field="y",
                                                                    batch_size=1)

    def test_pytorch_model(self):
        steps = [ProteinStandardizer(), Word2Vec()]
        model = TestPytorchBaselineModel(512, 50)
        pytorch_model = PyTorchModel(
            model=model,
            loss_function=nn.BCELoss(),
            device="cpu",
            validation_metric=accuracy_score,
            problem_type=BINARY, batch_size=2, epochs=2,
            optimizer=Adam(model.parameters(), lr=0.0001), progress=50,
            logger_path="small_dataset.log"
        )
        for step in steps:
            step.fit_transform(self.single_input_dataset)
            step.fit_transform(self.single_input_dataset_val)
            self.assertTrue(step.fitted)

        pytorch_model.fit(self.single_input_dataset, self.single_input_dataset_val)

        probs = pytorch_model.predict_proba(self.single_input_dataset)
        self.assertTrue(probs.shape[0] == 3)

    def test_tensorflow_model(self):
        steps = [ProteinStandardizer(), Word2Vec()]
        model = ToyTensorflowModel(512)
        model = TensorflowModel(model.model, problem_type=BINARY, epochs=1, batch_size=10)

        for step in steps:
            step.fit_transform(self.single_input_dataset)
            self.assertTrue(step.fitted)

        model.fit(self.single_input_dataset, self.single_input_dataset)
        self.assertTrue(model.predict_proba(self.single_input_dataset).shape[0] == 3)
