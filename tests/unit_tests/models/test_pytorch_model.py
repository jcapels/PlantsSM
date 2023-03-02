import os

from tests import TEST_DIR

from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam

from plants_sm.models.constants import BINARY
from plants_sm.models.pytorch_model import PyTorchModel
from unit_tests.models._utils import TestPytorchBaselineModel
from unit_tests.models.test_models import TestModels


class TestPytorchModel(TestModels):

    def test_pytorch_linear_model(self):
        model = TestPytorchBaselineModel(100, 50)
        pytorch_model = PyTorchModel(model=model,
                                     loss_function=nn.BCELoss(),
                                     device="cpu",
                                     validation_metric=accuracy_score,
                                     problem_type=BINARY, batch_size=2, epochs=1,
                                     optimizer=Adam(model.parameters(), lr=0.0001), progress=50,
                                     logger_path="small_dataset.log"
                                     )

        pytorch_model.fit(self.train_dataset, self.validation_dataset)
        predictions = pytorch_model.predict(self.validation_dataset)
        self.assertEqual(predictions.shape, self.validation_dataset.y.shape)

        predictions_proba = pytorch_model.predict_proba(self.validation_dataset)
        self.assertEqual(predictions_proba.shape, self.validation_dataset.y.shape)

    def test_save_load_model(self):
        model = TestPytorchBaselineModel(100, 50)
        pytorch_model = PyTorchModel(model=model,
                                     loss_function=nn.BCELoss(),
                                     device="cpu",
                                     validation_metric=accuracy_score,
                                     problem_type=BINARY, batch_size=2, epochs=1,
                                     optimizer=Adam(model.parameters(), lr=0.0001), progress=50,
                                     logger_path="small_dataset.log"
                                     )

        pytorch_model.fit(self.train_dataset, self.validation_dataset)
        pytorch_model.save(os.path.join(TEST_DIR, "data", "test_save_model"))
        pytorch_model = PyTorchModel.load(self.path_to_save)
        predictions = pytorch_model.predict(self.validation_dataset)
        self.assertEqual(predictions.shape, self.validation_dataset.y.shape)

        predictions_proba = pytorch_model.predict_proba(self.validation_dataset)
        self.assertEqual(predictions_proba.shape, self.validation_dataset.y.shape)

        self.assertEqual(pytorch_model.problem_type, BINARY)
        self.assertEqual(pytorch_model.epochs, 1)
        self.assertEqual(pytorch_model.batch_size, 2)
        self.assertEqual(pytorch_model.optimizer.__class__.__name__, "Adam")
        self.assertEqual(pytorch_model.loss_function.__class__.__name__, "BCELoss")
        self.assertEqual(pytorch_model.validation_metric, accuracy_score)
        self.assertEqual(pytorch_model.progress, 50)
