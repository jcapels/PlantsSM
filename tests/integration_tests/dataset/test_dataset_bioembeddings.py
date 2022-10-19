import pandas as pd

from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_structures.dataset import SingleInputDataset
from plants_sm.featurization.proteins.bio_embeddings.plus_rnn_embedding import PlusRNNEmbedding
from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert
from plants_sm.featurization.proteins.bio_embeddings.unirep import UniRepEmbeddings
from plants_sm.featurization.proteins.bio_embeddings.word2vec import Word2Vec


class TestDatasetBioembeddings(TestDataset):

    def test_create_dataset_bioembeddings(self):
        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence")
        unirep = UniRepEmbeddings()
        unirep.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 1900)
        self.assertEqual(dataset.features_fields[0], "unirep_1")
        self.assertEqual(dataset.instances_ids_field, "identifier")

        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence",
                                     instances_ids_field="ids")

        unirep = UniRepEmbeddings(output_shape_dimension=3)
        unirep.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 454)
        self.assertEqual(dataset.X.shape[2], 1900)
        self.assertEqual(dataset.features_fields[0], "unirep_1")
        self.assertEqual(dataset.instances_ids_field, "ids")

    def test_word2vec_dataset(self):
        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence")
        word2vec = Word2Vec()
        word2vec.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 512)
        self.assertEqual(dataset.features_fields[0], "word2vec_1")
        self.assertEqual(dataset.instances_ids_field, "identifier")

        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence")
        word2vec = Word2Vec(output_shape_dimension=3)
        word2vec.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 453)
        self.assertEqual(dataset.X.shape[2], 512)
        self.assertEqual(dataset.features_fields[0], "word2vec_1")
        self.assertEqual(dataset.instances_ids_field, "identifier")

    def test_multithreading_features_extraction(self):
        dataframe = pd.DataFrame(columns=["ids", "sequence"])
        sequences = [
            "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
            "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
            "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPWSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
            "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
            "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
            "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDWSFILK"
        ]
        dataframe["sequence"] = sequences
        dataframe["ids"] = ["WP_003399671.1"]

        dataset = SingleInputDataset(dataframe=dataframe, representation_field="sequence",
                                     instances_ids_field="ids")

        unirep = UniRepEmbeddings()
        unirep.fit_transform(dataset)

        dataset2 = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence",
                                      instances_ids_field="ids")

        unirep = UniRepEmbeddings(n_jobs=2)
        unirep.fit_transform(dataset2)

    def test_create_dataset_prot_bert(self):
        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence")
        unirep = ProtBert()
        unirep.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 1024)
        self.assertEqual(dataset.features_fields[0], "prot_bert_1")
        self.assertEqual(dataset.instances_ids_field, "identifier")

        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence",
                                     instances_ids_field="ids")

        unirep = ProtBert(output_shape_dimension=3)
        unirep.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 453)
        self.assertEqual(dataset.X.shape[2], 1024)
        self.assertEqual(dataset.features_fields[0], "prot_bert_1")
        self.assertEqual(dataset.instances_ids_field, "ids")

    def test_create_dataset_plus_rnn(self):
        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence")
        unirep = PlusRNNEmbedding(n_jobs=2, device="cpu")
        unirep.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 1024)
        self.assertEqual(dataset.features_fields[0], "plus_rnn_1")
        self.assertEqual(dataset.instances_ids_field, "identifier")

        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence")
        unirep = PlusRNNEmbedding(n_jobs=2, output_shape_dimension=3, device="cpu")
        unirep.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 453)
        self.assertEqual(dataset.X.shape[2], 1024)
        self.assertEqual(dataset.features_fields[0], "plus_rnn_1")
        self.assertEqual(dataset.instances_ids_field, "identifier")
