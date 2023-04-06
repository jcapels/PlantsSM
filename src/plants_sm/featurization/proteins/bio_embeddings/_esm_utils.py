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

    def __init__(self, backend="NCCL", num_machines=1, machine_rank=0, dist_url="tcp://127.0.0.1:1234") -> None:
        self.backend = backend
        # self.num_gpus = num_gpus
        self.num_machines = num_machines
        self.machine_rank = machine_rank
        self.dist_url = dist_url

    def run(self, main_func, **kwargs):
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["NCCL_DEBUG"] = "ERROR"

        world_size = 1

        results_file = tempfile.NamedTemporaryFile(delete=True)

        mp.spawn(
            TorchSpawner.distributed_worker,
            nprocs=1,
            args=(
                main_func,
                self.backend,
                world_size,
                1,
                self.machine_rank,
                self.dist_url,
                results_file.name,
                kwargs
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
        results_file,
        kwargs
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
                timeout=DEFAULT_TIMEOUT,
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

        results = main_func(**kwargs)
        with open(results_file, "wb") as results_file:
            pickle.dump(results, results_file)