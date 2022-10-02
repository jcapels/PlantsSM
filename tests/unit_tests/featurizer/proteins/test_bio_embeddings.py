import os

from yaml import YAMLError

from plants_sm.featurization.proteins.bio_embeddings._utils import get_model_file, get_device, read_config_file
from plants_sm.featurization.proteins.bio_embeddings.word2vec import Word2Vec
from ...featurizer.proteins.test_protein_featurizers import TestProteinFeaturizers

from plants_sm.featurization.proteins.bio_embeddings.unirep import UniRepEmbeddings
from tests import TEST_DIR


class TestEmbeddings(TestProteinFeaturizers):

    def test_unirep_embeddings(self):
        dataset = UniRepEmbeddings().fit_transform(self.dataset)
        self.assertEqual(dataset.features_dataframe.shape, (2, 1901))
        self.assertAlmostEqual(0.016243484, dataset.features_dataframe.iloc[0, 1])

    def test_unirep_embeddings_3d(self):
        dataset = UniRepEmbeddings(output_shape_dimension=3).fit_transform(self.dataset)

        features = dataset.features_dataframe.to_numpy() \
            .reshape((len(dataset.identifiers), dataset.features_shape[1], len(dataset.features_fields)))

        self.assertEqual((2, 454, 1900), features.shape)

    def test_raise_errors_unirep(self):
        with self.assertRaises(NotImplementedError):
            UniRepEmbeddings(output_shape_dimension=2, device="2").fit_transform(self.dataset)

    def test_word2vec_embeddings_2d(self):
        dataset = Word2Vec().fit_transform(self.dataset)
        self.assertEqual(dataset.features_dataframe.shape, (2, 513))
        self.assertAlmostEqual(-0.033814143, dataset.features_dataframe.iloc[0, 1])

    def test_word2vec_embeddings_3d(self):
        dataset = Word2Vec(output_shape_dimension=3).fit_transform(self.dataset)

        features = dataset.features_dataframe.to_numpy() \
            .reshape((len(dataset.identifiers), dataset.features_shape[1], len(dataset.features_fields)))

        self.assertEqual(dataset.dataframe.shape, (2, 2))
        self.assertEqual(features.shape, (2, 453, 512))

    def test_get_model_function(self):
        self.assertIn("plants_sm/word2vec/model_file", get_model_file("word2vec", "model_file"))

    def test_get_device(self):
        device = get_device("cpu")
        self.assertEqual(device.type, "cpu")

        device = get_device("cuda")
        self.assertEqual(device.type, "cuda")

        device = get_device()
        self.assertEqual(device.type, "cuda")

    def test_read_config_file(self):
        config_file = os.path.join(TEST_DIR, "data", "defaults.yml")
        config = read_config_file(config_file)

        self.assertEqual(config["word2vec"]["model_file"],
                         "http://data.bioembeddings.com/public/embeddings/embedding_models/word2vec/word2vec.model")

        with self.assertRaises(YAMLError):
            config_file = os.path.join(TEST_DIR, "data", "defaults_error.yml")
            read_config_file(config_file)