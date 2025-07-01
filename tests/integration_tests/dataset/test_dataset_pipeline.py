from integration_tests.dataset.test_dataset import TestDataset
import numpy as np
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam

from plants_sm.ml.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.ml.data_structures.dataset import PLACEHOLDER_FIELD, SingleInputDataset
from plants_sm.ml.featurization.proteins.propythia.propythia import PropythiaWrapper
from plants_sm.ml.models.constants import BINARY
from plants_sm.ml.models.pytorch_model import PyTorchModel
from plants_sm.ml.pipeline.pipeline import Pipeline
from unit_tests.models._utils import TestPytorchBaselineModel


class TestDatasetPipeline(TestDataset):

    def test_dataset_pipeline(self):
        steps = [ProteinStandardizer(), PropythiaWrapper(preset='performance')]
        model = TestPytorchBaselineModel(8676, 50)
        pytorch_model = PyTorchModel(model=model,
                                     loss_function=nn.BCELoss(),
                                     device="cpu",
                                     validation_metric=accuracy_score,
                                     problem_type=BINARY, batch_size=2, epochs=2,
                                     optimizer=Adam(model.parameters(), lr=0.0001), progress=50,
                                     logger_path="small_dataset.log"
                                     )

        pipeline = Pipeline(steps, models=[pytorch_model])

        with self.assertRaises(AssertionError):
            pipeline.fit(self.single_input_dataset, self.single_input_dataset)

        pipeline.fit(self.single_input_dataset, self.single_input_dataset_val)
        for step in pipeline.steps[PLACEHOLDER_FIELD]:
            self.assertTrue(step.fitted)

        probs = pipeline.predict_proba(self.single_input_dataset)

        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(probs.shape[0], self.single_input_dataset.y.shape[0])

        probs = pipeline.predict(self.single_input_dataset)

        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(probs.shape[0], 3)

    def test_pipeline_to_train_in_batch(self):

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

        steps = [ProteinStandardizer(), PropythiaWrapper(preset='performance')]
        model = TestPytorchBaselineModel(8676, 50)
        pytorch_model = PyTorchModel(model=model,
                                     loss_function=nn.BCELoss(),
                                     device="cpu",
                                     validation_metric=accuracy_score,
                                     problem_type=BINARY, batch_size=2, epochs=2,
                                     optimizer=Adam(model.parameters(), lr=0.0001), progress=50
                                     )

        pipeline = Pipeline(steps, models=[pytorch_model])

        with self.assertRaises(AssertionError):
            pipeline.fit(self.single_input_dataset, self.single_input_dataset)

        pipeline.fit(self.single_input_dataset, self.single_input_dataset_val)
        for step in pipeline.steps[PLACEHOLDER_FIELD]:
            self.assertTrue(step.fitted)

        probs = pipeline.predict_proba(self.single_input_dataset_val)

        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(probs.shape[0], 3)

        probs = pipeline.predict(self.single_input_dataset)

        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(probs.shape[0], 3)
