from typing import Any

import optuna
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam

from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.featurization.proteins.bio_embeddings.word2vec import Word2Vec
from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper
from plants_sm.hyperparameter_optimization.experiment import Experiment
from plants_sm.models.constants import BINARY
from plants_sm.models.pytorch_model import PyTorchModel
from plants_sm.pipeline.pipeline import Pipeline
from unit_tests.models._utils import TestPytorchBaselineModel


class MockExperiment(Experiment):

    def objective(self, trial: optuna.trial.Trial) -> float:
        lr = trial.suggest_float("lr", 0.0001, 0.1, log=True)
        intermediate_dim = trial.suggest_int("intermediate_dim", 20, 500)

        pipeline = self.pipeline_runner(lr, intermediate_dim)
        predictions = pipeline.predict(self.validation_dataset)
        return accuracy_score(self.validation_dataset.y, predictions)

    def pipeline_runner(self, lr: float, intermediate_dim: int):
        steps = [ProteinStandardizer(), PropythiaWrapper(preset='performance')]
        model = TestPytorchBaselineModel(8677, 50, intermediate_dim=intermediate_dim)
        pytorch_model = PyTorchModel(model=model,
                                     loss_function=nn.BCELoss(),
                                     device="cpu",
                                     validation_metric=accuracy_score,
                                     problem_type=BINARY, batch_size=2, epochs=2,
                                     optimizer=Adam(model.parameters(), lr=lr), progress=50,
                                     logger_path="small_dataset.log"
                                     )
        pipeline = Pipeline(steps, models=[pytorch_model])
        pipeline.fit(self.train_dataset, self.validation_dataset)
        return pipeline

    @property
    def best_experiment(self) -> Any:
        return self.pipeline_runner(**self.best_parameters)
