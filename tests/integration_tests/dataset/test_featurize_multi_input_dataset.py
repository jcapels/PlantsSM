from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors
from plants_sm.featurization.proteins.bio_embeddings.unirep import UniRepEmbeddings
from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper


class TestFeaturizeMultiInputDataset(TestDataset):

    def test_propythia_deepmol_featurizer(self):
        PropythiaWrapper(preset="all", n_jobs=2).fit_transform(self.multi_input_dataset, "proteins")
        DeepMolDescriptors().fit_transform(self.multi_input_dataset, "ligands")

        self.assertEqual(self.multi_input_dataset.X["proteins"].shape[1], 9595)
        self.assertEqual(self.multi_input_dataset.X["ligands"].shape[1], 2048)

    def test_unirep_deepmol_featurizer(self):
        UniRepEmbeddings(n_jobs=2).fit_transform(self.multi_input_dataset, "proteins")
        DeepMolDescriptors().fit_transform(self.multi_input_dataset, "ligands")

        self.assertEqual(self.multi_input_dataset.X["proteins"].shape[1], 1900)
