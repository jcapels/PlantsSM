import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from fairscale.nn.wrap import enable_wrap, wrap

import esm

from typing import Union
import torch
from esm.model.esm2 import ESM2


DEFAULT_TIMEOUT = timedelta(seconds=10)


def run(main_func, backend, num_machines, num_gpus, machine_rank, dist_url, is_ddf, esm_model, esm_num_gpus):
    world_size = num_machines * num_gpus

    mp.spawn(
        distributed_worker,
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
            esm_num_gpus
        ),
        daemon=False,
    )

    # main_func(args)

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

    main_func(is_ddf, esm_model, esm_num_gpus)


def main(backend="NCCL", num_gpus=1, num_machines=1, machine_rank=0, dist_url="tcp://127.0.0.1:1234", 
         is_ddf = True, esm_model = esm.pretrained.esm2_t33_650M_UR50D):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NCCL_DEBUG"] = "ERROR"

    print(f"CUDA {torch.version.cuda} - cuDNN {torch.backends.cudnn.version()} - cudaNCCL {torch.cuda.nccl.version()}")

    if is_ddf:
        run(
            main_func=generate_esm,
            backend=backend,
            num_machines=num_machines,
            num_gpus=1,
            machine_rank=machine_rank,
            dist_url=dist_url,
            is_ddf=is_ddf,
            esm_model=esm_model,
            esm_num_gpus=num_gpus
        )
    else:
        generate_esm(is_ddf, esm_model)


class ESMModel(ESM2):

    def __init__(self,  num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True, is_ddp = False,
        num_gpus=1) -> None:
        self.is_ddp = is_ddp
        self.num_gpus = num_gpus
        super().__init__(num_layers, embed_dim, attention_heads, alphabet, token_dropout)

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        if self.is_ddp:
            gpus = list(range(torch.cuda.device_count()))
            if len(gpus) >= self.num_gpus:
                gpus = gpus[:self.num_gpus]
            gpus = [f"cuda:{i}" for i in range(len(gpus))]
        
        i = 0
        i += 1
        for layer_idx, layer in enumerate(self.layers):
            if self.is_ddp:
                gpu = gpus[i % len(gpus)]
                x.to(gpu)
                layer.to(gpu)
            else:
                x = x.to("cpu")
                layer = layer.to("cpu")

            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )

            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))
            i += 1

        if self.is_ddp:
            gpu = gpus[i % len(gpus)]
            x.to(gpu)
            self.emb_layer_norm_after.to(gpu)
        else:
            x = x.to("cpu")
            self.emb_layer_norm_after = self.emb_layer_norm_after.to("cpu")

        i+=1
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        if self.is_ddp:
            gpu = gpus[i % len(gpus)]
            x.to(gpu)
            self.lm_head.to(gpu)
        else:
            x = x.to("cpu")
            self.lm_head = self.lm_head.to("cpu")
        i+=1
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result


def generate_esm(is_ddf, esm_model, num_gpus=1):

    data = [
        ("protein1", """MAGIIKKQILKHLSRFTKNLSPDKINLSTLKGEGELKNLELDEEVLQNMLDLPTWLAINK
VFCNKASIRIPWTKLKTHPICLSLDKVIMEMSTCEEPRSPNGPSPIATASGQSEYGFAEK
VVEGISVSVNSIVIRIGAKAFNASFELSQLRIYSVNAHWEHGDLRFTRIQDPQRGEVLTF
KEINWQMIRIEADATQSSHLEIMCAPVRLITNQSKIRVTLKRRLKDCNVIATKLVLILDD
LLWVLTDSQLKAMVQYAKSLSEAIEKSTEQRKSMAPEPTQSSTVVASAQQVKTTQTSNAP
DVNDAIVKLFNDFDVKETSHHLVISHLDLHICDDIHAKEKESNRRITGGAMQLSFTQLTI
DYYPYHKAGDSCNHWMYFSDATKTKNGWANELLHEFECNVEMLKQAVKDHNVGSPPKSPT
HASPQHTQTEKDYPLKGTCRTPSVLSQQSKAKLMSSSVVVRLADFNIYQVSTAEQCRSSP
KSMICCNKKSLYLPQEMSAVYIEFTEYYYPDGKDFPIPSPNLYSQLNALQFTVDERSILW
LNQFLLDLKQSLNQFMAVYKLNDNSKSDEHVDVRVDGLMLKFVIPSEVKSECHQDQPRAI
SIQSSEMIATNTRHCPNCRHSDLEALFQDFKDCDFFSKTYTSFPKSCDNFNLLHPIFQRH
AHEQDTKMHEIYKGNITPQLNKNTLKTSAATDVWAVYFSQFWIDYEGMKSGKGRPISFVD
SFPLSIWICQPTRYAESQKEPQTCNQVSLNTSQSESSDLAGRLKRKKLLKEYYSTESEPL
TNGGQKPSSSDTFFRFSPSSSEADIHLLVHVHKHVSMQINHYQYLLLLFLHESLILLSEN
LRKDVEAVTGSPASQTSICIGILLRSAELALLLHPVDQANTLKSPVSESVSPVVPDYLPT
ENGDFLSSKRKQISRDINRIRSVTVNHMSDNRSMSVDLSHIPLKDPLLFKSASDTNLQKG
ISFMDYLSDKHLGKISEDESSGLVYKSGSGEIGSETSDKKDSFYTDSSSILNYREDSNIL
SFDSDGNQNILSSTLTSKGNETIESIFKAEDLLPEAASLSENLDISKEETPPVRTLKSQS
SLSGKPKERCPPNLAPLCVSYKNMKRSSSQMSLDTISLDSMILEEQLLESDGSDSHMFLE
KGNKKNSTTNYRGTAESVNAGANLQNYGETSPDAISTNSEGAQENHDDLMSVVVFKITGV
NGEIDIRGEDTEICLQVNQVTPDQLGNISLRHYLCNRPVGSDQKAVIHSKSSPEISLRFE
SGPGAVIHSLLAEKNGFLQCHIENFSTEFLTSSLMNIQHFLEDETVATVMPMKIQVSNTK
INLKDDSPRSSTVSLEPAPVTVHIDHLVVERSDDGSFHIRDSHMLNTGNDLKENVKSDSV
LLTSGKYDLKKQRSVTQATQTSPGVPWPSQSANFPEFSFDFTREQLMEENESLKQELAKA
KMALAEAHLEKDALLHHIKKMTVE""")
    ]

    for i in range(1, 30):
        data.append(("protein{}".format(i + 1), data[0][1]))

    ESM_LAYERS = {
    "esm1_t6_43M_UR50S": 6,
    "esm1_t12_85M_UR50S": 12,
    "esm1_t34_670M_UR100": 34,
    "esm1_t34_670M_UR50D": 34,
    "esm1_t34_670M_UR50S": 34,
    "esm1b_t33_650M_UR50S": 33,
    "esm_msa1_t12_100M_UR50S": 12,
    "esm_msa1b_t12_100M_UR50S": 12,
    "esm1v_t33_650M_UR90S_1": 33,
    "esm1v_t33_650M_UR90S_2": 33,
    "esm1v_t33_650M_UR90S_3": 33,
    "esm1v_t33_650M_UR90S_4": 33,
    "esm1v_t33_650M_UR90S_5": 33,
    "esm_if1_gvp4_t16_142M_UR50": 16,
    "esm2_t6_8M_UR50D": 6,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
    "esm2_t36_3B_UR50D": 36,
    "esm2_t48_15B_UR50D": 48,

}
    
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

    start = time.time()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    if is_ddf:
        batch_tokens = batch_tokens.cuda()
    else:
        batch_tokens = batch_tokens.to("cpu")
        model = model.to("cpu")

    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[ESM_LAYERS[esm_model.__name__]], return_contacts=False)

    token_representations = results["representations"][ESM_LAYERS[esm_model.__name__]]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    batch_lens = (batch_tokens != vocab.padding_idx).sum(1)
    

    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    end = time.time()
    print("Time taken 10 sequences: ", end - start)


if __name__ == '__main__':
    from esm.pretrained import esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, \
        esm2_t36_3B_UR50D, esm2_t48_15B_UR50D
    
    esm_models = [esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, \
        esm2_t36_3B_UR50D]
    esm_models=[esm2_t33_650M_UR50D]
    for model in esm_models:
        
        print("")
        print("Model: ", model.__name__)

        main(is_ddf=True, esm_model=model, num_gpus=2)
        print()

        # print("In CPU")

        # main(is_ddf=False, esm_model=model)
        # print()
