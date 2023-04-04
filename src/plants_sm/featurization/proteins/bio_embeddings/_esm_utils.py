import os
import pickle
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

import esm

import torch
from plants_sm.featurization.proteins.bio_embeddings._esm_model import ESMModel

import tempfile


DEFAULT_TIMEOUT = timedelta(seconds=10)

class TorchSpawner:

    @staticmethod
    def run(main_func, backend, num_machines, num_gpus, machine_rank, dist_url, is_ddf, esm_model, esm_num_gpus, data, num_layers):
        world_size = num_machines * num_gpus

        results_file = tempfile.NamedTemporaryFile(delete=True)

        mp.spawn(
            TorchSpawner.distributed_worker,
            nprocs=num_gpus,
            args=(
                main_func,
                backend,
                world_size,
                num_gpus,
                machine_rank,
                dist_url,
                is_ddf,
                esm_model,
                esm_num_gpus,
                data,
                num_layers,
                results_file.name
            ),
            daemon=False,
        )

        results = pickle.load(results_file)
        results_file.close()
        return results


    @staticmethod
    def distributed_worker(
        local_rank,
        main_func,
        backend,
        world_size,
        num_gpus_per_machine,
        machine_rank,
        dist_url,
        is_ddf,
        esm_model,
        esm_num_gpus,
        data,
        num_layers,
        results_file,
        timeout=DEFAULT_TIMEOUT,
    ):
        LOCAL_PROCESS_GROUP = None

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your installation.")

        global_rank = machine_rank * num_gpus_per_machine + local_rank
        try:
            dist.init_process_group(
                backend=backend,
                init_method=dist_url,
                world_size=world_size,
                rank=global_rank,
                timeout=timeout,
            )
        except Exception as e:
            raise e


        dist.barrier()


        if num_gpus_per_machine > torch.cuda.device_count():
            raise RuntimeError
        torch.cuda.set_device(local_rank)

        # Setup the local process group (which contains ranks within the same machine)
        if LOCAL_PROCESS_GROUP is not None:
            raise RuntimeError

        num_machines = world_size // num_gpus_per_machine

        for idx in range(num_machines):
            ranks_on_i = list(range(idx * num_gpus_per_machine, (idx + 1) * num_gpus_per_machine))
            pg = dist.new_group(ranks_on_i)
            if idx == machine_rank:
                LOCAL_PROCESS_GROUP = pg

        results = main_func(data, is_ddf, esm_model, num_layers, esm_num_gpus)
        with open(results_file, "wb") as results_file:
            pickle.dump(results, results_file)
        

    @staticmethod
    def generate_esm(data, num_layers, backend="NCCL", num_gpus=1, num_machines=1, machine_rank=0, dist_url="tcp://127.0.0.1:1234", 
            is_ddf = True, esm_model = esm.pretrained.esm2_t33_650M_UR50D):
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["NCCL_DEBUG"] = "ERROR"

        print(f"CUDA {torch.version.cuda} - cuDNN {torch.backends.cudnn.version()} - cudaNCCL {torch.cuda.nccl.version()}")

        if is_ddf:
            results = TorchSpawner.run(
                main_func=_generate_esm,
                backend=backend,
                num_machines=num_machines,
                num_gpus=1,
                machine_rank=machine_rank,
                dist_url=dist_url,
                is_ddf=is_ddf,
                esm_model=esm_model,
                esm_num_gpus=num_gpus, 
                data=data, 
                num_layers=num_layers
            )
            return results
        else:
            return _generate_esm(data, is_ddf, esm_model, num_layers)


def _generate_esm(data, is_ddf, esm_model, num_layers, num_gpus=1):

    # _, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

    if is_ddf:
        cpu_offload = False
    else:
        cpu_offload = True
    
    # initialize the model with FSDP wrapper
    fsdp_params = dict(
        mixed_precision=True,
        flatten_parameters=True,
        state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
        cpu_offload=cpu_offload,  # enable cpu offloading
    )

    if is_ddf:

        with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
            model, vocab = esm_model()
            batch_converter = vocab.get_batch_converter()
            model.eval()

            ddp_model = ESMModel(alphabet=vocab, num_layers=model.num_layers, embed_dim=model.embed_dim, 
                                attention_heads=model.attention_heads, token_dropout=model.token_dropout, 
                                is_ddp=is_ddf, num_gpus=num_gpus)
            ddp_model.load_state_dict(model.state_dict())
            model = ddp_model

            # Wrap each layer in FSDP separately
            for name, child in model.named_children():
                
                if name == "layers":
                    for layer_name, layer in child.named_children():
                        wrapped_layer = wrap(layer)
                        setattr(child, layer_name, wrapped_layer)

            model = wrap(model)
    else:
        model, vocab = esm_model()
        batch_converter = vocab.get_batch_converter()
        model.eval()

        ddp_model = ESMModel(alphabet=vocab, num_layers=model.num_layers, embed_dim=model.embed_dim, 
                            attention_heads=model.attention_heads, token_dropout=model.token_dropout, 
                            is_ddp=is_ddf)
        ddp_model.load_state_dict(model.state_dict())
        model = ddp_model

    # batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    _, _, batch_tokens = batch_converter(data)
    if is_ddf:
        batch_tokens = batch_tokens.cuda()
    else:
        batch_tokens = batch_tokens.to("cpu")
        model = model.to("cpu")
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[num_layers], return_contacts=False)

    print(results["representations"])
    return results