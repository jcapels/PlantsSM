from plants_sm.featurization.proteins.bio_embeddings.esm_models import ESM1Model, ESM2Model
from plants_sm.parallelisation import TorchSpawner
from torch import nn
from tqdm import tqdm

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization._utils import call_set_features_names
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_DIMENSIONS, ESM_FUNCTIONS, ESM_LAYERS
from plants_sm.transformation.transformer import Transformer

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

import torch

from esm.pretrained import load_model_and_alphabet_local


class ESMEncoder(Transformer):
    """
    It encodes protein sequences with the embedding layer of the pre-trained model ESM-1B.
    The EsmEncoder operates only over pandas DataFrame.

    Parameters
    ----------
    batch_size: int, optional (default=16)
        The batch size to be used in the encoding process. Higher batch sizes can lead to OOM issues.
    esm_function: str, optional (default="esm1b_t33_650M_UR50S")
        The ESM function to be used.
    device: str, optional (default="cpu")
        The device to be used for the encoding process.
    num_gpus: int, optional (default=None)
        The number of GPUs to be used for the encoding process.
    output_dim: int, optional (default=2)
        The output dimension of the model.
    """

    batch_size: int = 16
    features_names = list = []
    esm_function: str = "esm2_t6_8M_UR50D"
    device: str = "cpu"
    num_gpus: int = None
    output_dim: int = 2
    return_contacts: bool = False
    model_path: str = None

    def set_features_names(self):
        """
        Set the features names of the encoded dataset.
        """
        self.features_names = [f"ESM_{self.esm_function}_{i}" for i in range(ESM_DIMENSIONS[self.esm_function])]

    def _fit(self, dataset: Dataset, instance_type: str) -> 'ESMEncoder':
        """
        Fit the ESM. It loads the pre-trained model and the batch converter.

        Parameters
        ----------
        dataset: Dataset
            The dataset to be used to fit the Esm1bEncoder.
        instance_type: str
            The type of instance to be used to fit the ESM.

        Returns
        -------
        encoder: a fitted ESM
        """

        if self.esm_function in ESM_DIMENSIONS:

            esm_callable = ESM_FUNCTIONS[self.esm_function]
            self.layers = ESM_LAYERS[self.esm_function]

            if self.model_path is not None:
                model, self.alphabet = load_model_and_alphabet_local(self.model_path)
            else:
                model, self.alphabet = esm_callable()
            
            if self.num_gpus is not None and self.device == "cpu":
                self.device = "cuda"
                
            self.model = model
            self.batch_converter = self.alphabet.get_batch_converter()

            if self.num_gpus is not None and self.num_gpus > 1:
                self.is_ddf = True
            elif self.num_gpus is not None and self.num_gpus == 1:
                self.is_ddf = False
            else:
                self.num_gpus = 0
                self.is_ddf = False

            return self
        else:
            raise ValueError(f"Invalid esm_function. Available functions are: {list(ESM_DIMENSIONS.keys())}")

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'ESMEncoder':

        return self._fit(dataset, instance_type)

    @staticmethod
    def _generate_esm2_model(model: nn.Module,
                             layers: int,
                             instances: dict,
                             batch_size: int,
                             batch_converter: callable,
                             output_dim: int,
                             is_ddf: bool,
                             device: str = "cpu"):
        """
        Generate the ESM model.

        Parameters
        ----------
        model: nn.Module
            The ESM model.
        layers: int
            The number of layers of the ESM model.
        instances: dict
            The instances to be encoded.
        batch_size: int
            The batch size to be used in the encoding process.
        batch_converter: callable
            The batch converter to be used in the encoding process.
        output_dim: int
            The output dimension of the ESM model.
        is_ddf: bool
            Whether to use DDP or not.
        """

        if is_ddf:
            fsdp_params = dict(
                mixed_precision=True,
                flatten_parameters=True,
                state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
                cpu_offload=False,  # enable cpu offloading
            )

            with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
                model.eval()

                # Wrap each layer in FSDP separately
                for name, child in model.named_children():

                    if name == "layers":
                        for layer_name, layer in child.named_children():
                            wrapped_layer = wrap(layer)
                            setattr(child, layer_name, wrapped_layer)

                model = wrap(model)

        res = []
        batch = []
        batch_ids = []

        pbar = tqdm(desc="ESM", total=len(instances.items()))
        for instance_id, instance_representation in instances.items():

            batch.append((instance_id, instance_representation))
            batch_ids.append(instance_id)
            if len(batch) == batch_size:
                representations = {}
                _, _, batch_tokens = batch_converter(batch)
                if is_ddf:
                    batch_tokens = batch_tokens.cuda()
                else:
                    batch_tokens = batch_tokens.to(device)

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[layers], return_contacts=False)
                    results["representations"][layers] = results["representations"][layers].cpu().detach().numpy()
                    representations['representations'] = results["representations"][layers]

                    for i, batch_instance_id in enumerate(batch_ids):

                        if output_dim == 2:
                            res.append((batch_instance_id,
                                        representations['representations'][i, 1: len(batch[i][1]) + 1].mean(0)))
                        else:
                            res.append((batch_instance_id,
                                        representations['representations'][i, 1: len(batch[i][1]) + 1]))
                    if is_ddf:
                        torch.cuda.empty_cache()
                    batch = []
                    batch_ids = []
                    pbar.update(batch_size)

        if len(batch) != 0:

            representations = {}
            _, _, batch_tokens = batch_converter(batch)

            if is_ddf:
                batch_tokens = batch_tokens.cuda()
            else:
                batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[layers], return_contacts=False)
            results["representations"][layers] = results["representations"][layers].cpu().detach().numpy()
            representations['representations'] = results["representations"][layers]

            for i, batch_instance_id in enumerate(batch_ids):
                if output_dim == 2:
                    res.append((batch_instance_id,
                                representations['representations'][i, 1: len(batch[i][1]) + 1].mean(0)))
                else:
                    res.append((batch_instance_id,
                                representations['representations'][i, 1: len(batch[i][1]) + 1]))

            torch.cuda.empty_cache()
            pbar.update(len(batch_ids))

        return res

    @call_set_features_names
    def _transform(self, dataset: Dataset, instance_type: str) -> Dataset:
        """
        It encodes a protein sequence with the embedding layer of the pre-trained model ESM-1B.

        Parameters
        ----------
        dataset: Dataset
            The dataset to be used to encode the protein sequences.
        instance_type: str
            The instance type to be encoded.

        Returns
        -------
        encoded_sequence: np.ndarray
            The encoded protein sequence.
        """

        instances = dataset.get_instances(instance_type)

        # initialize the model with FSDP wrapper
        if "esm2" in self.esm_function:
            if self.num_gpus == 1 and self.device != "cpu":
                devices = [self.device]
            else:
                devices = None

            model = ESM2Model(alphabet=self.alphabet, num_layers=self.model.num_layers, embed_dim=self.model.embed_dim,
                              attention_heads=self.model.attention_heads, token_dropout=self.model.token_dropout,
                              is_ddf=self.is_ddf, num_gpus=self.num_gpus, devices=devices)
            model.load_state_dict(self.model.state_dict())

        else:
            model = ESM1Model(self.model.args, alphabet=self.alphabet,
                               device=self.device)
            model.load_state_dict(self.model.state_dict())

        if self.is_ddf:
            res = TorchSpawner().run(self._generate_esm2_model,
                                     model=model,
                                     layers=self.layers,
                                     instances=instances,
                                     batch_size=self.batch_size,
                                     batch_converter=self.batch_converter,
                                     output_dim=self.output_dim,
                                     is_ddf=self.is_ddf,
                                     device=self.device)

        else:
            res = self._generate_esm2_model(model,
                                            layers=self.layers,
                                            instances=instances,
                                            batch_size=self.batch_size,
                                            batch_converter=self.batch_converter,
                                            output_dim=self.output_dim,
                                            is_ddf=self.is_ddf,
                                            device=self.device)

        dataset.add_features(instance_type, dict(res))

        dataset.features_fields[instance_type] = self.features_names

        return dataset
