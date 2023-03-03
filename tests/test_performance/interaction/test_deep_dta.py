import os
import sys
from unittest import TestCase, skip

import tensorflow as tf

from plants_sm.data_standardization.compounds.deepmol_standardizers import DeepMolStandardizer
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.featurization.encoding.label_encoder import LabelEncoder
from plants_sm.models.constants import BINARY
from plants_sm.models.interaction.deep_dta import DeepDTATensorflow
from plants_sm.models.tensorflow_model import TensorflowModel
from plants_sm.tokenisation.compounds.smilespe import AtomLevelTokenizer
from tests import TEST_DIR

@skip
class TestDeepDTA(TestCase):

    def setUp(self) -> None:
        csv_to_read = os.path.join(TEST_DIR, "compound_protein_interaction", "super_train.csv")
        self.dataset_35000_instances_train = MultiInputDataset.from_csv(csv_to_read,
                                                                        representation_fields={"proteins": "SEQ",
                                                                                               "ligands": "SUBSTRATES"},
                                                                        instances_ids_field={"interaction": "index"},
                                                                        labels_field="activity")

        csv_to_read = os.path.join(TEST_DIR, "compound_protein_interaction", "super_valid.csv")
        self.dataset_35000_instances_valid = MultiInputDataset.from_csv(csv_to_read,
                                                                        representation_fields={"proteins": "SEQ",
                                                                                               "ligands": "SUBSTRATES"},
                                                                        instances_ids_field={"interaction": "index"},
                                                                        labels_field="activity")

        environment_name = sys.executable.split('/')[-3]
        print('Environment:', environment_name)
        os.environ[environment_name] = str(123)
        os.environ['PYTHONHASHSEED'] = str(123)
        os.environ['TF_DETERMINISTIC_OPS'] = 'False'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        os.environ['LD_LIBRARY_PATH'] = sys.executable.replace('bin/python', 'lib/')
        env_path = sys.executable.replace('bin/python', 'lib')
        # cdll.LoadLibrary(f'{env_path}/libcublas.so.11')
        # cdll.LoadLibrary(f'{env_path}/libcudart.so.11.0')
        # cdll.LoadLibrary(f'{env_path}/libcublasLt.so.11')
        # cdll.LoadLibrary(f'{env_path}/libcufft.so.10')
        # cdll.LoadLibrary(f'{env_path}/libcurand.so.10')
        # cdll.LoadLibrary(f'{env_path}/libcusolver.so.11')
        # cdll.LoadLibrary(f'{env_path}/libcusparse.so.11')
        # cdll.LoadLibrary(f'{env_path}/libcudnn.so.7')

        tf.random.set_seed(123)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        G = tf.Graph()
        session = tf.compat.v1.Session(graph=G,
                                       config=config)

        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # print('GPUs:', gpus)
        # if gpus:
        #     # Restrict TensorFlow to only use the first GPU
        #     try:
        #         tf.config.experimental.set_visible_devices(gpus, 'GPU')
        #     except RuntimeError as e:
        #         # Visible devices must be set at program startup
        #         print(e)

    def test_deep_dta(self):
        HEAVY_STANDARDIZATION = {
            'remove_isotope'.upper(): True,
            'NEUTRALISE_CHARGE'.upper(): True,
            'remove_stereo'.upper(): True,
            'keep_biggest'.upper(): True,
            'add_hydrogen'.upper(): True,
            'kekulize'.upper(): False,
            'neutralise_charge_late'.upper(): True
        }

        kwargs = {"params": HEAVY_STANDARDIZATION}

        DeepMolStandardizer(preset="custom_standardizer", kwargs=kwargs, n_jobs=8).fit_transform(
            self.dataset_35000_instances_train,
            "ligands")

        ProteinStandardizer(n_jobs=8).fit_transform(self.dataset_35000_instances_valid, "proteins")

        DeepMolStandardizer(preset="custom_standardizer", kwargs=kwargs, n_jobs=8).fit_transform(
            self.dataset_35000_instances_valid,
            "ligands")

        ProteinStandardizer(n_jobs=8).fit_transform(self.dataset_35000_instances_train, "proteins")

        one_hot = LabelEncoder(alphabet="ARNDCEQGHILKMFPSTWYV").fit(
            self.dataset_35000_instances_train,
            "proteins")
        one_hot.transform(self.dataset_35000_instances_train,
                          "proteins")
        one_hot.transform(self.dataset_35000_instances_valid,
                          "proteins")

        one_hot_compounds = LabelEncoder(tokenizer=AtomLevelTokenizer()).fit(
            self.dataset_35000_instances_train,
            "ligands")
        one_hot_compounds.transform(self.dataset_35000_instances_train, "ligands")

        one_hot_compounds.transform(self.dataset_35000_instances_valid, "ligands")

        input_size_proteins = self.dataset_35000_instances_train.X["proteins"].shape
        input_size_compounds = self.dataset_35000_instances_train.X["ligands"].shape

        n_char_proteins = 21
        n_char_compounds = len(one_hot_compounds.tokens) + 1
        model = DeepDTATensorflow(input_size_proteins[1], input_size_compounds[1], n_char_proteins, n_char_compounds,
                                  16)

        wrapper = TensorflowModel(model=model.model, epochs=50, batch_size=32, problem_type=BINARY)
        wrapper.fit(self.dataset_35000_instances_train, self.dataset_35000_instances_valid)
