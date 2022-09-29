from ...featurizer.proteins.test_protein_featurizers import TestProteinFeaturizers

from plants_sm.featurization.proteins.bio_embeddings.unirep import UniRepEmbeddings


class TestEmbeddings(TestProteinFeaturizers):

    def test_unirep_embeddings(self):
        dataset = UniRepEmbeddings().fit_transform(self.dataset)
        self.assertEqual(dataset.dataframe.shape, (2, 1900))
