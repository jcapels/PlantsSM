import numpy as np
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam

from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_structures.dataset import PLACEHOLDER_FIELD
from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper
from plants_sm.models.constants import BINARY
from plants_sm.models.pytorch_model import PyTorchModel
from plants_sm.pipeline.pipeline import Pipeline
from unit_tests.models._utils import TestPytorchBaselineModel
from unit_tests.models.test_models import TestModels


class TestPipeline(TestModels):

    def test_fit_predict(self):

        steps = [ProteinStandardizer(), PropythiaWrapper(preset='performance')]
        model = TestPytorchBaselineModel(100, 50)
        pytorch_model = PyTorchModel(model=model,
                                     loss_function=nn.BCELoss(),
                                     device="cpu",
                                     validation_metric=accuracy_score,
                                     problem_type=BINARY, batch_size=2, epochs=2,
                                     optimizer=Adam(model.parameters(), lr=0.0001), progress=50,
                                     logger_path="small_dataset.log"
                                     )

        pipeline = Pipeline(steps=steps, models=[pytorch_model])

        pipeline.fit(self.train_dataset, self.validation_dataset)
        for step in pipeline.steps[PLACEHOLDER_FIELD]:
            self.assertTrue(step.fitted)

        probs = pipeline.predict_proba(self.validation_dataset)

        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(probs.shape[0], self.train_dataset.y.shape[0])

        probs = pipeline.predict(self.validation_dataset)

        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(probs.shape[0], self.validation_dataset.y.shape[0])

    def test_load_save(self):

        steps = [ProteinStandardizer(), PropythiaWrapper(preset='performance')]
        model = TestPytorchBaselineModel(100, 50)
        pytorch_model = PyTorchModel(model=model,
                                     loss_function=nn.BCELoss(),
                                     device="cpu",
                                     validation_metric=accuracy_score,
                                     problem_type=BINARY, batch_size=2, epochs=2,
                                     optimizer=Adam(model.parameters(), lr=0.0001), progress=50,
                                     logger_path="small_dataset.log"
                                     )

        pipeline = Pipeline(steps, models=[pytorch_model])
        pipeline.fit(self.train_dataset, self.validation_dataset)

        # create temp file
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            probs_before = pipeline.predict(self.validation_dataset)
            pipeline.save(tmp)
            new_pipeline = Pipeline.load(tmp)
            probs_after = new_pipeline.predict(self.validation_dataset)
            self.assertIsInstance(probs_after, np.ndarray)
            self.assertEqual(probs_after.shape[0], self.train_dataset.y.shape[0])
            for i in range(len(probs_after)):
                self.assertEqual(probs_after[i], probs_before[i])
