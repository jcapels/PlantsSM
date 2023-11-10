from torch import nn

from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_FUNCTIONS, ESM_LAYERS
from plants_sm.featurization.proteins.bio_embeddings.esm_models import ESM1Model
from plants_sm.models.fc.fc import DNN


class EC_ESM1b(nn.Module):
    def __init__(self, hidden_layers, num_classes, is_ddf, num_gpus):
        super(EC_ESM1b, self).__init__()
        esm_callable = ESM_FUNCTIONS["esm1b_t33_650M_UR50S"]
        self.layers = ESM_LAYERS["esm1b_t33_650M_UR50S"]
        self.is_ddf = is_ddf
        self.num_gpus = num_gpus

        model, self.alphabet = esm_callable()

        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model = ESM1Model(model.args, alphabet=self.alphabet,
                                   is_ddf=self.is_ddf, num_gpus=self.num_gpus)

        self.esm_model.load_state_dict(model.state_dict())

        self.dnn = DNN(1280, hidden_layers, num_classes, batch_norm=True, last_sigmoid=True)

    def forward(self, data):

        output = self.esm_model(data, repr_layers=[self.layers])
        output = output["representations"][self.layers]
        # if use_cuda:
        #     output = output.cuda(gpu, non_blocking=True)
        output = output[:, 0, :]
        # if use_cuda:
        #     x = x.cuda(gpu, non_blocking=True)

        x = self.dnn(output)
        return x
