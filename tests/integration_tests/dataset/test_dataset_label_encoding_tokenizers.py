from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.ml.featurization.encoding.label_encoder import LabelEncoder
from plants_sm.ml.tokenisation.compounds.smilespe import AtomLevelTokenizer


class TestDatasetLabelEncoderTokenizer(TestDataset):

    def test_dataset_label_encoder_tokenizer(self):
        tokenizer = LabelEncoder(tokenizer=AtomLevelTokenizer())
        tokenizer.fit_transform(self.multi_input_dataset, instance_type="ligands")

        self.assertEqual(self.multi_input_dataset.dataframe.shape[0], 9)
        self.assertIn("[C@@H]", self.multi_input_dataset.features_fields["ligands"])
        self.assertEqual(self.multi_input_dataset.instances_ids_field["interaction"], "ids")
