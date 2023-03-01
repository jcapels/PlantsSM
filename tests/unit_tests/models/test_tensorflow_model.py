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
        self.assertEqual(predictions.shape, (self.validation_dataset.y.shape[0], ))
