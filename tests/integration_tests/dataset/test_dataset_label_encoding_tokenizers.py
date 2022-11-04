from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD
from plants_sm.featurization.encoding.label_encoder import LabelEncoder
from plants_sm.tokenisation.compounds.smilespe import AtomLevelTokenizer


class TestDatasetLabelEncoderTokenizer(TestDataset):

    def test_dataset_label_encoder_tokenizer(self):
        tokenizer = LabelEncoder(tokenizer=AtomLevelTokenizer())
        tokenizer.fit_transform(self.multi_input_dataset, instance_type="ligands")

        self.assertEqual(self.multi_input_dataset.dataframe.shape[0], 2)
        self.assertEqual(self.multi_input_dataset.X.shape[1], 3)
        self.assertEqual(self.multi_input_dataset.features_fields[PLACEHOLDER_FIELD][0], "label_encoder_tokenizer_1")
        self.assertEqual(self.multi_input_dataset.instances_ids_field, "identifier")
