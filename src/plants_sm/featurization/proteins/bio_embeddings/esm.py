import torch
from tqdm import tqdm

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization._utils import call_set_features_names
from plants_sm.featurization.proteins.bio_embeddings._esm_utils import TorchSpawner
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_DIMENSIONS, ESM_FUNCTIONS, ESM_LAYERS
from plants_sm.transformation.transformer import Transformer

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

import torch
from plants_sm.featurization.proteins.bio_embeddings._esm_model import ESMModel


class ESMEncoder(Transformer):
    """
    It encodes protein sequences with the embedding layer of the pre-trained model ESM-1B.
    The Esm1bEncoder operates only over pandas DataFrame.

    Parameters
    ----------
    batch_size: int, optional (default=16)
        The batch size to be used in the encoding process. Higher batch sizes can lead to OOM issues.

    """

    batch_size: int = 16
    features_names = list = []
    esm_function: str = "esm2_t6_8M_UR50D"
    device: str = "cpu"
    num_gpus: int = None 
    output_dim: int = 2
    return_contacts: bool = False

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

            model, self.alphabet = esm_callable()
            self.model = model.to(self.device)
            self.batch_converter = self.alphabet.get_batch_converter()

            if self.num_gpus is not None:
                self.is_ddf = True
            else:
                self.num_gpus = 0
                self.is_ddf = False

            return self
        else:
            raise ValueError(f"Invalid esm_function. Available functions are: {list(ESM_DIMENSIONS.keys())}")
    
    @staticmethod
    def _generate_esm_model(model, layers, instances, 
                            batch_size, batch_converter, 
                            output_dim, num_gpus, 
                            alphabet, is_ddf):

        if is_ddf:
            fsdp_params = dict(
                mixed_precision=True,
                flatten_parameters=True,
                state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
                cpu_offload=False,  # enable cpu offloading
            )

            with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
                model.eval()

                ddp_model = ESMModel(alphabet=alphabet, num_layers=model.num_layers, embed_dim=model.embed_dim, 
                                    attention_heads=model.attention_heads, token_dropout=model.token_dropout, 
                                    is_ddp=True, num_gpus=num_gpus)
                ddp_model.load_state_dict(model.state_dict())
                model = ddp_model

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
                    batch_tokens = batch_tokens.to("cpu")
                
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[layers], return_contacts=False)
                    representations['representations'] = results["representations"][layers].cpu().detach().numpy()

                    for i, batch_instance_id in enumerate(batch_ids):
                        if output_dim == 2:
                            res.append((batch_instance_id,
                                    representations['representations'][i, 1: len(batch[i][1]) + 1].mean(0)))
                        else:
                            res.append((batch_instance_id,
                                    representations['representations'][i, 1: len(batch[i][1]) + 1]))

                    batch = []
                    batch_ids = []
                    pbar.update(batch_size)

        if len(batch) != 0:

            representations = {}
            _, _, batch_tokens = batch_converter(batch)

            if is_ddf:
                batch_tokens = batch_tokens.cuda()
            else:
                batch_tokens = batch_tokens.to("cpu")

            results = model(batch_tokens, repr_layers=[layers], return_contacts=False)
            representations['representations'] = results["representations"][layers].cpu().detach().numpy()

            for i, batch_instance_id in enumerate(batch_ids):
                if output_dim == 2:
                    res.append((batch_instance_id,
                            representations['representations'][i, 1: len(batch[i][1]) + 1].mean(0)))
                else:
                    res.append((batch_instance_id,
                            representations['representations'][i, 1: len(batch[i][1]) + 1]))
            
            pbar.update(len(batch_ids))
            batch = []
            batch_ids = []

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
        # it has to run in batch of 16, otherwise can lead to OOM issues

        res = []
        instances = dataset.get_instances(instance_type)

        # initialize the model with FSDP wrapper

        if self.is_ddf:
            res = TorchSpawner().run(self._generate_esm_model, 
                                        model = self.model, 
                                        layers = self.layers, 
                                        instances= instances, 
                                        batch_size=self.batch_size, 
                                        batch_converter = self.batch_converter, 
                                        output_dim=self.output_dim, 
                                        num_gpus=self.num_gpus,
                                        alphabet=self.alphabet,
                                        is_ddf=self.is_ddf)
            
        else:
            res = self._generate_esm_model(self.model,
                                        layers = self.layers, 
                                        instances= instances, 
                                        batch_size=self.batch_size, 
                                        batch_converter = self.batch_converter, 
                                        output_dim=self.output_dim, 
                                        num_gpus=self.num_gpus,
                                        alphabet=self.alphabet,
                                        is_ddf=self.is_ddf)

        dataset.features[instance_type] = dict(res)

        if instance_type not in dataset.features_fields:
            dataset.features_fields[instance_type] = self.features_names
        else:
            dataset.features_fields[instance_type].extend(self.features_names)

        return dataset
