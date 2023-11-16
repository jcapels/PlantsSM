

import torch
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset


from plants_sm.data_standardization.truncation import Truncator
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_LAYERS

from plants_sm.models.ec_number_prediction.esm import EC_ESM1b_Lightning, EC_ESM_Lightning

from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import EarlyStopping, BaseFinetuning

from sklearn.metrics import f1_score

import lightning as L


def f1_score_macro(y_pred, y_true):
    return f1_score(y_pred, y_true, average="macro", zero_division=1)

class FreezeUnfreezeModules(BaseFinetuning):
    def __init__(self, modules, unfreeze_at_epoch=10):
        super().__init__()
        self.modules = modules
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, module):
        # freeze any module you want
        for name, parameter in module.named_parameters():
            if name in self.modules:
                parameter.requires_grad = False

    def finetune_function(self, module, current_epoch, optimizer):
        # When `current_epoch` is 10, feature_extractor will start training.
        # if current_epoch == self._unfreeze_at_epoch:
        #     self.unfreeze_and_add_param_group(
        #         modules=module,
        #         optimizer=optimizer,
        #         train_bn=True,
        #     )
        pass

if __name__=="__main__":
    # train_dataset = SingleInputDataset.from_csv("../../final_data/train.csv", representation_field="sequence", instances_ids_field="accession", labels_field=slice(8, 2779))
    # train_dataset = Truncator(max_length=884).fit_transform(train_dataset)

    # validation_dataset = SingleInputDataset.from_csv("../../final_data/validation.csv", representation_field="sequence", instances_ids_field="accession", labels_field=slice(8, 2779))
    # validation_dataset = Truncator(max_length=884).fit_transform(validation_dataset)

    sharding_strategy="FULL_SHARD"
    limit_all_gathers=True
    cpu_offload=True
    callbacks = EarlyStopping("val_metric", patience=5, mode="max")

    strategy = FSDPStrategy(sharding_strategy=sharding_strategy, limit_all_gathers=limit_all_gathers, cpu_offload=cpu_offload)
    

    tensor_train_dataset = torch.load('/home/jcapela/PlantsSM/examples/tensor_train_dataset.pt', map_location=torch.device('cpu'))
    tensor_validation_dataset = torch.load('/home/jcapela/PlantsSM/examples/tensor_validation_dataset.pt', map_location=torch.device('cpu'))

    model = EC_ESM_Lightning("esm2_t12_35M_UR50D", [2560, 5120], 2771, batch_size=2)
    layers_1 = ESM_LAYERS["esm2_t12_35M_UR50D"] - 1
    layer_2 = ESM_LAYERS["esm2_t12_35M_UR50D"] - 2
    no_grad = set()
    for parameter in model.named_parameters():
        if "layers" in parameter[0] and str(layers_1) not in parameter[0] and str(layer_2) not in parameter[0]:
            no_grad.add(parameter[0])
    model = EC_ESM_Lightning("esm2_t12_35M_UR50D", [2560, 5120], 2771, metric=f1_score_macro,
                                max_epochs=2,
                                devices=[0, 1, 2, 3 ,4, 5, 6, 7], 
                                batch_size=4, strategy="fsdp",
                                accelerator="gpu", 
                                callbacks=[callbacks],
                                no_grad=no_grad)
    model.fit(tensor_train_dataset, validation_dataset=tensor_validation_dataset)
    
    # model.fit(train_dataset, validation_dataset=validation_dataset)
    # model.save("test_model")

    # model = EC_ESM_Lightning.load_from_checkpoint("/home/jcapela/PlantsSM/examples/test_model/pytorch_model_weights.ckpt",
    #                                             model_name="esm2_t12_35M_UR50D", hidden_layers=[2560, 5120], num_classes=2771, 
    #                                             batch_size=16)
    # model = EC_ESM_Lightning.load("/home/jcapela/PlantsSM/examples/test_model/")
    # print(model.predict(dataset, trainer = L.Trainer(accelerator="cuda")))


    # model = EC_ESM_Lightning("esm2_t12_35M_UR50D", [2560, 5120], 2771, batch_size=2)
    # tensor_train_dataset = model.preprocess(train_dataset)
    # torch.save(tensor_train_dataset, 'tensor_train_dataset.pt')

    # tensor_validation_dataset = model.preprocess(validation_dataset)
    # torch.save(tensor_validation_dataset, 'tensor_validation_dataset.pt')