from copy import copy

import pandas as pd

from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD
from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert
from plants_sm.featurization.proteins.bio_embeddings.unirep import UniRepEmbeddings


class TestDatasetBioembeddings(TestDataset):

    def test_create_dataset_bioembeddings(self):
        dataframe = copy(self.dataframe)
        dataset = SingleInputDataset(dataframe=dataframe, representation_field="sequence")
        unirep = UniRepEmbeddings()
        unirep.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 1900)
        self.assertEqual(dataset.features_fields[PLACEHOLDER_FIELD][0], "unirep_1")
        self.assertEqual(dataset.instances_ids_field, "identifier")

        dataframe = copy(self.dataframe)
        dataset_2 = SingleInputDataset(dataframe=dataframe, representation_field="sequence",
                                       instances_ids_field="ids")

        unirep = UniRepEmbeddings(output_shape_dimension=3)
        unirep.fit_transform(dataset_2)

        self.assertEqual(dataset_2.X.shape[0], 2)
        self.assertEqual(dataset_2.X.shape[1], 454)
        self.assertEqual(dataset_2.X.shape[2], 1900)
        self.assertEqual(dataset_2.features_fields[PLACEHOLDER_FIELD][0], "unirep_1")
        self.assertEqual(dataset_2.instances_ids_field, "ids")

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
        self.assertEqual(dataset.features_fields[PLACEHOLDER_FIELD][0], "prot_bert_1")
        self.assertEqual(dataset.instances_ids_field, "identifier")

        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence",
                                     instances_ids_field="ids")

        unirep = ProtBert(output_shape_dimension=3)
        unirep.fit_transform(dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.X.shape[1], 453)
        self.assertEqual(dataset.X.shape[2], 1024)
        self.assertEqual(dataset.features_fields[PLACEHOLDER_FIELD][0], "prot_bert_1")
        self.assertEqual(dataset.instances_ids_field, "ids")
