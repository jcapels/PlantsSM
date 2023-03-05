from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam

from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.featurization.proteins.bio_embeddings.word2vec import Word2Vec
from plants_sm.models.constants import BINARY
from plants_sm.models.pytorch_model import PyTorchModel
from plants_sm.pipeline.pipeline import Pipeline
from unit_tests.models._utils import TestPytorchBaselineModel
from unit_tests.models.test_models import TestModels


class TestPipeline(TestModels):

    def test_fit(self):

        steps = [ProteinStandardizer(), Word2Vec()]
        model = TestPytorchBaselineModel(100, 50)
        pytorch_model = PyTorchModel(model=model,
                                     loss_function=nn.BCELoss(),
                                     device="cuda",
                                     validation_metric=accuracy_score,
                                     problem_type=BINARY, batch_size=2, epochs=1,
                                     optimizer=Adam(model.parameters(), lr=0.0001), progress=50,
                                     logger_path="small_dataset.log"
                                     )

        Pipeline(steps, models=[pytorch_model]).fit(self.train_dataset, self.validation_dataset)
