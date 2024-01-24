import os
from unittest import skip

from yaml import YAMLError

from plants_sm.featurization.proteins.bio_embeddings._utils import get_model_file, get_device, read_config_file
from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert
from plants_sm.featurization.proteins.bio_embeddings.word2vec import Word2Vec
from tests.unit_tests.featurizer.proteins.test_protein_featurizers import TestProteinFeaturizers

from plants_sm.featurization.proteins.bio_embeddings.unirep import UniRepEmbeddings
from tests import TEST_DIR


class TestEmbeddings(TestProteinFeaturizers):

    def test_unirep_embeddings(self):
        UniRepEmbeddings().fit_transform(self.dataset)
        self.assertEqual(self.dataset.X().shape, (2, 1900))
        self.assertAlmostEqual(0.016247222, self.dataset.X()[0, 0], delta=0.0005)

    def test_unirep_embeddings_3d(self):
        UniRepEmbeddings(output_shape_dimension=3).fit_transform(self.dataset)

        self.assertEqual((2, 454, 1900), self.dataset.X().shape)

    def test_raise_errors_unirep(self):
        UniRepEmbeddings(output_shape_dimension=2, device="2").fit_transform(self.dataset)

    def test_prot_bert_embeddings(self):
        ProtBert().fit_transform(self.dataset)
        self.assertEqual(self.dataset.X().shape, (2, 1024))
        self.assertAlmostEqual(0.11255153, self.dataset.X()[0, 0], delta=0.005)

    def test_prot_bert_embeddings_3d(self):
        ProtBert(output_shape_dimension=3).fit_transform(self.dataset)
        self.assertEqual(self.dataset.X().shape, (2, 453, 1024))
        self.assertAlmostEqual(-0.006767738, self.dataset.X()[0, 0, 0], delta=0.005)

    @skip("No memory on CI")
    def test_esm_1b(self):
        ESMEncoder(device="cpu", esm_function="esm1b_t33_650M_UR50S", batch_size=1).fit_transform(self.dataset)

    @skip("No memory on CI")
    def test_esm_2(self):
        ESMEncoder(device="cuda", esm_function="esm2_t6_8M_UR50D", batch_size=2, num_gpus=3).fit_transform(self.dataset)
        self.assertEqual(self.dataset.X().shape, (2, 320))
        self.assertAlmostEqual(-0.014742036, self.dataset.X()[0, 0], delta=0.005)

    def test_get_model_function(self):
        self.assertIn("plants_sm/word2vec/model_file", get_model_file("word2vec", "model_file"))

    def test_get_device(self):
        device = get_device("cpu")
        self.assertEqual(device.type, "cpu")

        device = get_device()
        self.assertEqual(device.type, "cpu")

    def test_read_config_file(self):
        config_file = os.path.join(TEST_DIR, "data", "defaults.yml")
        config = read_config_file(config_file)

        self.assertEqual(config["word2vec"]["model_file"],
                         "http://data.bioembeddings.com/public/embeddings/embedding_models/word2vec/word2vec.model")

        with self.assertRaises(YAMLError):
            config_file = os.path.join(TEST_DIR, "data", "defaults_error.yml")
            read_config_file(config_file)
