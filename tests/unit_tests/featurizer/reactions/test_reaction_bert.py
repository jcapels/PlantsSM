from unittest import skip
import numpy as np
from plants_sm.featurization.reactions.reaction_bert import ReactionBERT
from tests.unit_tests.featurizer.reactions.test_reaction_featurizers import TestReactionFeaturizers

# TODO: change the model (it has to be imported from transformers)
@skip
class TestReactionEmbeddings(TestReactionFeaturizers):

    def test_reaction_bert(self):
        ReactionBERT(bert_model_path="/home/jcapela/plants_ec_number_prediction/PlantsSM/tests/data/smiles_reaction_bert_model.pt").fit_transform(self.dataset)
        self.assertEqual(self.dataset.X().shape, (2, 2400))
        self.assertNotEqual(np.sum(self.dataset.X()), 0)