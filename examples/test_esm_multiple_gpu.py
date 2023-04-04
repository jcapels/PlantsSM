import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

import esm

# init the distributed world with world_size 1
def init_distributed():
    # url = "tcp://localhost:23456"
    # torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)
    # if torch.distributed.is_initialized():
    #     print("Process group initialized successfully!")
    # else:
    #     print("Process group failed to initialize.")

    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
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
HYVLPQLRKHHPWMWISH""")
    ]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    print(sequence_representations)

init_distributed()