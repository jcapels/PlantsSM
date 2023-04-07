import argparse
import os
from typing import Union
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
import torch
import esm

import torch
from esm.model.esm2 import ESM2


DEFAULT_TIMEOUT = timedelta(seconds=10)

from torch.nn.parallel import DistributedDataParallel as DDP

### Step 1: setup and cleanup setups
def setup(
    backend,
    world_size,
    num_gpus,
    machine_rank,
    dist_url,
    timeout=DEFAULT_TIMEOUT):

    global_rank = machine_rank * num_gpus
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=1,
        rank=global_rank,
        timeout=timeout,
    )

def cleanup():
    dist.destroy_process_group()

class ESMModel(ESM2):

    def __init__(self,  num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,) -> None:
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

        
        gpus = list(range(torch.cuda.device_count()))
        gpus = [f"cuda:{i}" for i in range(len(gpus))]
        i = 0
        for layer_idx, layer in enumerate(self.layers):
            gpu = gpus[i % len(gpus)]
            x.to(gpu)
            layer.to(gpu)
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

        gpu = gpus[i % len(gpus)]
        x.to(gpu)
        self.emb_layer_norm_after.to(gpu)
        i+=1
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        gpu = gpus[i % len(gpus)]
        x.to(gpu)
        self.lm_head.to(gpu)
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


def init(
    backend,
    world_size,
    num_gpus,
    machine_rank,
    dist_url,
    timeout=DEFAULT_TIMEOUT,
    args=(),):

    setup(backend,
            world_size,
            num_gpus,
            machine_rank,
            dist_url,
            timeout, 
            )


    model, vocab = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = vocab.get_batch_converter()
    model.eval()

    ddp_model = ESMModel(alphabet=vocab, num_layers=model.num_layers, embed_dim=model.embed_dim, 
                            attention_heads=model.attention_heads, token_dropout=model.token_dropout)
    ddp_model.load_state_dict(model.state_dict())
    
    ddp_model = ddp_model.cuda()

    ddp_model = DDP(ddp_model)

    ddp_model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        # ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        # ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3", """MVSQVLQLLRQGVWAALTGGWYHDPEQSKFTNSCHLYLWLFLLLLPLALHLAFPPNAIIV
FFYCSAVTIFFTIIKLVSYRLHLMFDKGEVIQQKPSRKEEKPNKDKEAKGEHITNHRNPS
NNRQIHNGKKEEASRNLSTPPLRCSSRGQSITSHHSSGPLELSAQETVEDLKGVILLEDH
PIAPVSSTSPGIKVESLPASQAHMLETTTKSVIPVKPVATETLINGKGKERGGKGQPPLR
HRSEGGLVDKGPLKKLPHLSLSQYDLLETDVSFQPWGSENSVLIPEPVSCPRGSIRERVQ
SKSPQDSLSSSCPQCDTIVAKPVEEPADTSCQVDTSCQGDLPLHQEVDSSDSEVAVTLID
TSQPGDPLSLHEPIKIVITMSSTPNSMTDLESSLHLRVVGTEKTSVKSDAEPTNPGAAGS
PNAEQISIPVITLDLPEGGGGGVPCPEGNGSERTPERLKTRVSTNQCSGYGSGEGGNAIK
DHSSSSREPWESVSRLTPDTGSESKVGKEGQTNLDPSSCKSSHEKRHARVLSVDSGTDVF
LSKSSAEIVNDTEKTMPTSKSDLEAKEGQMPNESNFLEFVSLLESINTSKMTASSQLNGS
AEQNEESGLLRDNCSQEKKEEILENEKPSGHSSKQGKPDLQSQDHTSTGPACTQPAKTTA
FFQGNRQRQIIYRVTSQQDSSVLQVISGPETSVQEEISVDAMHVFIDEHGEIRSCYLKSG
NQKEGPLQPLPSNNDCLSQAREMQVSSSSTTTSESQDPSSGDPAVSALQQQLLLMVARRT
QSETPRHVSQDLEASSCSSTQGKFNREQFYKFIIFPGKWIKVWYDRLTLLALLDRTEDIK
ENVLAILLIVLVSLLGFLTLSQGFCKDMWVLLFCLVMASCQYSLLKSVQPDPASPIHGHN
QIITYSRPIYFCVLCGLILLLDTGAKARHPPSYVVYGLKLFSPVFLQSARDYLIVFLYCF
PAISLLGLFPQINTFCTYLLEQIDMLFFGGSAVSGITSAVYSVARSVLAAALLHAVCFSA
VKEPWSMQHIPALFSAFCGLLVALSYHLSRQSSDPSVLMSFIQCRLFPKFLHQNLAESAA
DPLPKKMKDSVTDVLKWDLIVCAVVAVLSFAVSASTVFLSLRPFLSIVLFALAGAVGFVT
HYVLPQLRKHHPWMWISHPILKNKEYHQREVRDVAHLMWFERLYVWLQCFEKYILYPALI
LNALTIDAFLISNHRRLGTHWDIFLMIIAGMKLLRTSFCNPVYQFINLSFTVIFFHFDYK
DISESFLLDFFMVSILFSKLGDLLHKLQFVLTYVAPWQMAWGSSFHVFAQLFAIPHSAML
FFQTIATSIFSTPLSPFLGSVIFITSYVRPVKFWEKNYNTRRVDNSNTRLAVQIERDPGN
DDNNLNSIFYEHLTRTLQESLCGDLVLGRWGNYSSGDCFILASDDLNAFVHLIEIGNGLV
TFQLRGLEFRGTYCQQREVEAIMEGDEEDRGCCCCKPGHLPHLLSCNAAFHLRWLTWEIT
QTQYILEGYSILDNNAATMLQVFDLRRILIRYYIKSIIYYMVTSPKLLSWIKNESLLKSL
QPFAKWHYIERDLAMFNINIDDDYVPCLQGITRASFCNVYLEWIQHCARKRQEPSTTLDS
DEDSPLVTLSFALCTLGRRALGTAAHNMAISLDSFLYGLHVLFKGDFRITARDEWVFADM
DLLHKVVAPAIRMSLKLHQDQFTCPDEYEDPAVLYEAIQSFEKKVVICHEGDPAWRGAVL
SNKEELLTLRHVVDEGADEYKVIMLHRSFLSFKVIKVNKECVRGLWAGQQQELIFLRNRN
PERGSIQNNKQVLRNLINSSCDQPLGYPMYVSPLTTSYLGTHRQLKNIWGGPITLDRIRT
WFWTKWVRMRKDCNARQHSGGNIEDVDGGGAPTTGGNNAPSGGSQESSAEQPRKGGAQHG
VSSCEGTQRTGRRKGRSQSVQAHSALSQRPPMLSSSGPILESRQTFLQTSTSVHELAQRL
SGSRLSLHASATSLHSQPPPVTTTGHLSVRERAEALIRSSLGSSTSSTLSFLFGKRSFSS
ALVISGLSAAEGGNTSDTQSSSSVNIVMGPSARAASQATRHLSEPCEPPDATEQGQLHDR
CLAEAVADTLGVVCRRASQEDMGLDDTASQQSVSDEQ""")
    ]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.cuda()
    # model = model.cuda()
    # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = ddp_model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    batch_lens = (batch_tokens != vocab.padding_idx).sum(1)
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    print(sequence_representations)

    cleanup()

def main():
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"

    print(f"CUDA {torch.version.cuda} - cuDNN {torch.backends.cudnn.version()} - cudaNCCL {torch.cuda.nccl.version()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="NCCL", help="'gloo' or 'NCCL'.")
    parser.add_argument("--num-gpus", type=int, default=1, help="# GPUs per machine.")
    parser.add_argument("--num-machines", type=int, default=1, help="# of machines.")
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine).",
    )

    port = 1234
    parser.add_argument(
        "--dist-url",
        type=str,
        default=f"tcp://127.0.0.1:{port}",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )

    args = parser.parse_args()

    world_size = args.num_machines * args.num_gpus

    init(
        backend=args.backend,
        world_size=world_size,
        num_gpus=args.num_gpus,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=()
    )

if __name__ == "__main__":
    main()