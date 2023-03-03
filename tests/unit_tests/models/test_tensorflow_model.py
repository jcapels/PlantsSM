from plants_sm.models.constants import BINARY
from plants_sm.models.tensorflow_model import TensorflowModel
from unit_tests.models._utils import ToyTensorflowModel
from unit_tests.models.test_models import TestModels


class TestTensorflowModel(TestModels):

    def test_model(self):
        model = ToyTensorflowModel(100)
        model = TensorflowModel(model.model, problem_type=BINARY, epochs=1, batch_size=10)
        model.fit(self.train_dataset, self.validation_dataset)
        predictions = model.predict(self.validation_dataset)
        self.assertEqual(predictions.shape[0], self.validation_dataset.y.shape[0])

        predictions_proba = model.predict_proba(self.validation_dataset)
        self.assertEqual(predictions_proba.shape[0], self.validation_dataset.y.shape[0])

        history = model.history.history
        self.assertIsInstance(history, dict)
        self.assertIn("loss", history)

    def test_save_load_model(self):
        model = ToyTensorflowModel(100)
        model = TensorflowModel(model.model, problem_type=BINARY, epochs=1, batch_size=10)
        model.fit(self.train_dataset, self.validation_dataset)
        model.save(self.path_to_save)

        model = TensorflowModel.load(self.path_to_save)
        predictions = model.predict(self.validation_dataset)
        self.assertEqual(predictions.shape[0], self.validation_dataset.y.shape[0])

        predictions_proba = model.predict_proba(self.validation_dataset)
        self.assertEqual(predictions_proba.shape[0], self.validation_dataset.y.shape[0])
        self.assertEqual(model.problem_type, BINARY)
        self.assertEqual(model.epochs, 1)
        self.assertEqual(model.batch_size, 10)
