import os
from unittest import TestCase

from sklearn.metrics import accuracy_score, precision_score
from torch import nn
from torch.optim import Adam

from plants_sm.ml.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.ml.data_structures.dataset import SingleInputDataset
from plants_sm.ml.featurization.proteins.propythia.propythia import PropythiaWrapper
from plants_sm.ml.models.constants import BINARY
from plants_sm.ml.models.pytorch_model import PyTorchModel
from tests import TEST_DIR
from unit_tests.models._utils import TestPytorchBaselineModel, DenseNet


class TestDatasetModel(TestCase):

    def setUp(self) -> None:
        self.single_input_dataset_csv = os.path.join(TEST_DIR, "data", "proteins.csv")
        self.multi_label_dataset_csv = os.path.join(TEST_DIR, "data", "multi_label_data.csv")

        self.single_input_dataset_batch = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                                                      representation_field="sequence",
                                                                      instances_ids_field="id",
                                                                      labels_field="y",
                                                                      batch_size=1)

        self.single_input_dataset_val_batch = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                                                          representation_field="sequence",
                                                                          instances_ids_field="id",
                                                                          labels_field="y",
                                                                          batch_size=1)

        self.single_input_dataset = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                                                representation_field="sequence",
                                                                instances_ids_field="id",
                                                                labels_field="y")

        self.single_input_dataset_val = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                                                    representation_field="sequence",
                                                                    instances_ids_field="id",
                                                                    labels_field="y")

        self.multi_label_dataset = SingleInputDataset.from_csv(self.multi_label_dataset_csv,
                                                               instances_ids_field="accession",
                                                               representation_field="sequence",
                                                               labels_field=slice(8, -1))

        self.multi_label_dataset_val = SingleInputDataset.from_csv(self.multi_label_dataset_csv,
                                                                   instances_ids_field="accession",
                                                                   representation_field="sequence",
                                                                   labels_field=slice(8, -1))

    def test_multi_label_dataset(self):
        steps = [ProteinStandardizer(), PropythiaWrapper(preset='performance')]

        model = DenseNet(8676, 256, 2895)

        def precision_average(y_true, y_pred):
            return precision_score(y_true, y_pred, average="macro")

        optimizer = Adam(params=model.parameters(), lr=0.001)
        model = PyTorchModel(batch_size=25, epochs=10,
                             loss_function=nn.BCEWithLogitsLoss(), optimizer=optimizer, model=model,
                             device="cpu", validation_metric=precision_average, problem_type=BINARY)
        for step in steps:
            step.fit_transform(self.multi_label_dataset)
            step.fit_transform(self.multi_label_dataset)
            step.fit_transform(self.multi_label_dataset_val)
            step.fit_transform(self.multi_label_dataset_val)
            self.assertTrue(step.fitted)

        model.fit(self.multi_label_dataset, self.multi_label_dataset_val)

        probs = model.predict(self.multi_label_dataset_val)
        self.assertTrue(probs.shape[1] == 2895)

    def test_pytorch_model_early_stopping_with_metric(self):
        steps = [ProteinStandardizer(), PropythiaWrapper(preset='performance')]

        model = DenseNet(8676, 256, 2895)

        def precision_average(y_true, y_pred):
            return precision_score(y_true, y_pred, average="macro")

        optimizer = Adam(params=model.parameters(), lr=0.001)
        model = PyTorchModel(batch_size=25, epochs=10,
                             loss_function=nn.BCEWithLogitsLoss(), optimizer=optimizer, model=model,
                             device="cpu", validation_metric=precision_average, problem_type=BINARY,
                             early_stopping_method="metric", objective="max")
        for step in steps:
            step.fit_transform(self.multi_label_dataset)
            step.fit_transform(self.multi_label_dataset)
            step.fit_transform(self.multi_label_dataset_val)
            step.fit_transform(self.multi_label_dataset_val)
            self.assertTrue(step.fitted)

        model.fit(self.multi_label_dataset, self.multi_label_dataset_val)

        probs = model.predict(self.multi_label_dataset_val)
        self.assertTrue(probs.shape[1] == 2895)

        print(precision_score(self.multi_label_dataset_val.y, probs, average="macro"))

    def test_pytorch_model(self):
        steps = [ProteinStandardizer(), PropythiaWrapper(preset='performance')]
        model = TestPytorchBaselineModel(8676, 50)
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
            step.fit_transform(self.single_input_dataset_batch)
            step.fit_transform(self.single_input_dataset_val_batch)
            self.assertTrue(step.fitted)

        pytorch_model.fit(self.single_input_dataset_batch, self.single_input_dataset_val_batch)

        probs = pytorch_model.predict(self.single_input_dataset_batch)
        self.assertTrue(probs.shape[0] == 3)
        ys = []
        while self.single_input_dataset_batch.next_batch():
            ys.extend(self.single_input_dataset_batch.y)

        probs = pytorch_model.predict(self.single_input_dataset_val_batch)
        self.assertTrue(probs.shape[0] == 3)
        ys = []
        while self.single_input_dataset_val_batch.next_batch():
            ys.extend(self.single_input_dataset_val_batch.y)

    def test_pytorch_model_with_custom_metric(self):

        def precision_average(y_true, y_pred):
            return precision_score(y_true, y_pred, average="macro")

        dataset = SingleInputDataset.from_csv(os.path.join(TEST_DIR, "data", "train_test.csv"),
                                              instances_ids_field="accession",
                                              representation_field="sequence",
                                              labels_field=slice(8, 2779))

        steps = [ProteinStandardizer(), PropythiaWrapper(preset='performance')]
        model = DenseNet(8676, 256, 2771)
        pytorch_model = PyTorchModel(
            model=model,
            loss_function=nn.BCELoss(),
            device="cpu",
            validation_metric=precision_average,
            problem_type=BINARY, epochs=4,
            optimizer=Adam(model.parameters(), lr=0.0001), progress=50,
            logger_path="small_dataset.log"
        )
        for step in steps:
            step.fit_transform(dataset)
            self.assertTrue(step.fitted)

        pytorch_model.fit(dataset)

        probs = pytorch_model.predict(dataset)

        print(precision_average(dataset.y, probs))

