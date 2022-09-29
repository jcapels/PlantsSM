import unittest

from plants_sm.featurization.proteins.bio_embeddings.word2vec import Word2Vec
from ...featurizer.proteins.test_protein_featurizers import TestProteinFeaturizers

from plants_sm.featurization.proteins.bio_embeddings.unirep import UniRepEmbeddings


class TestEmbeddings(TestProteinFeaturizers):

    def test_unirep_embeddings(self):
        dataset = UniRepEmbeddings().fit_transform(self.dataset)
        self.assertEqual(dataset.dataframe.shape, (2, 1902))
        self.assertEqual(0.016247222, dataset.dataframe.iloc[0, 2])

    # TODO: it generates 3-dimensional vectors, so it is not compatible with the current implementation
    # @unittest.skip("it generates 3-dimensional vectors, so it is not compatible with the current implementation")
    def test_word2vec_embeddings(self):
        dataset = Word2Vec().fit_transform(self.dataset)
        self.assertEqual(dataset.dataframe.shape, (2, 514))
        self.assertAlmostEqual(-0.033494536, dataset.dataframe.iloc[0, 2])
