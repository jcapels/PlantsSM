import os
from unittest import TestCase, skip

from plants_sm.data_standardization.compounds.deepmol_standardizers import DeepMolStandardizer
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors
from plants_sm.featurization.encoding.one_hot_encoder import OneHotEncoder
from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert
from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper

from tests import TEST_DIR


@skip("Just for performance testing")
class TestInteractionFeaturization(TestCase):
    """Test the interaction featurization performance."""

    def setUp(self) -> None:
        csv_to_read_400 = os.path.join(TEST_DIR, "performance_datasets", "aminotransferase_binary.csv")
        csv_to_read_35000 = os.path.join(TEST_DIR, "performance_datasets", "aminotransferase_binary.csv")
        self.multi_input_dataset_400 = MultiInputDataset.from_csv(csv_to_read_400,
                                                                  representation_fields={"proteins": "SEQ",
                                                                                         "ligands": "SUBSTRATES"},
                                                                  instances_ids_field={"interaction": "ids"},
                                                                  labels_field="LogSpActivity")

        self.multi_input_dataset_35000 = MultiInputDataset.from_csv(csv_to_read_35000,
                                                                    representation_fields={"proteins": "SEQ",
                                                                                           "ligands": "SUBSTRATES"},
                                                                    instances_ids_field={"interaction": "ids"},
                                                                    labels_field="LogSpActivity")

    def test_featurize_propythia_deepmol(self):
        """Test the featurization performance."""

        PropythiaWrapper(preset="all", n_jobs=8).fit_transform(self.multi_input_dataset_400, "proteins")
        DeepMolDescriptors(n_jobs=8).fit_transform(self.multi_input_dataset_400, "ligands")

        self.assertEqual(len(self.multi_input_dataset_400.X), 2)

    def test_featurize_one_hot_encodings_deepmol(self):
        """Test the featurization performance."""

        ProteinStandardizer(n_jobs=8).fit_transform(self.multi_input_dataset_400, "proteins")
        OneHotEncoder(n_jobs=8).fit_transform(self.multi_input_dataset_400, "proteins")
        DeepMolStandardizer(n_jobs=8, preset="custom_standardizer").fit_transform(self.multi_input_dataset_400,
                                                                                  "ligands")
        OneHotEncoder(n_jobs=8).fit_transform(self.multi_input_dataset_400, "ligands")

        self.assertEqual(len(self.multi_input_dataset_400.X), 2)

    def test_featurize_propythia_deepmol_35000(self):
        """Test the featurization performance."""

        PropythiaWrapper(preset="all", n_jobs=8).fit_transform(self.multi_input_dataset_35000, "proteins")
        DeepMolDescriptors(n_jobs=8).fit_transform(self.multi_input_dataset_35000, "ligands")

        self.assertEqual(len(self.multi_input_dataset_35000.X), 2)

    def test_featurize_protbert_deepmol(self):
        """Test the featurization performance."""

        ProtBert(n_jobs=1, device="cuda").fit_transform(self.multi_input_dataset_400, "proteins")
        DeepMolDescriptors(n_jobs=8).fit_transform(self.multi_input_dataset_400, "ligands")

        self.assertEqual(len(self.multi_input_dataset_400.X), 2)

    def test_featurize_protbert_deepmol_35000(self):
        """Test the featurization performance."""

        ProtBert(n_jobs=1, device="cuda").fit_transform(self.multi_input_dataset_35000, "proteins")
        DeepMolDescriptors(n_jobs=8).fit_transform(self.multi_input_dataset_35000, "ligands")

        self.assertEqual(len(self.multi_input_dataset_35000.X), 2)
