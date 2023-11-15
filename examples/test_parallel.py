

from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset


from plants_sm.data_standardization.truncation import Truncator

from plants_sm.models.ec_number_prediction.esm import EC_ESM1b_Lightning, EC_ESM_Lightning

from plants_sm.models.language_pytorch_models import EC_ESM1bLightningModel, EC_ESM2LightningModel
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import EarlyStopping

from sklearn.metrics import f1_score

import lightning as L

def f1_score_macro(y_pred, y_true):
    return f1_score(y_pred, y_true, average="macro", zero_division=1)

if __name__=="__main__":
    dataset = SingleInputDataset.from_csv("../../final_data/test_300.csv", representation_field="sequence", instances_ids_field="accession", labels_field=slice(8, 2779))
    dataset = Truncator(max_length=884).fit_transform(dataset)

    # sharding_strategy="FULL_SHARD"
    # limit_all_gathers=True
    # cpu_offload=True
    # callbacks = EarlyStopping("val_metric", patience=5, mode="max")

    # strategy = FSDPStrategy(sharding_strategy=sharding_strategy, limit_all_gathers=limit_all_gathers, cpu_offload=cpu_offload)
    
    # model = EC_ESM_Lightning("esm2_t12_35M_UR50D", [2560, 5120], 2771, metric=f1_score_macro,
    #                          max_epochs=2,
    #                             devices=[0, 1, 2], 
    #                             batch_size=4, strategy="fsdp",
    #                             accelerator="cuda", 
    #                             callbacks=[callbacks])
    # model.fit(dataset, dataset)
    # model.save("test_model")

    # model = EC_ESM_Lightning.load_from_checkpoint("/home/jcapela/PlantsSM/examples/test_model/pytorch_model_weights.ckpt",
    #                                             model_name="esm2_t12_35M_UR50D", hidden_layers=[2560, 5120], num_classes=2771, 
    #                                             batch_size=16)
    model = EC_ESM_Lightning.load("/home/jcapela/PlantsSM/examples/test_model/")
    print(model.predict(dataset, trainer = L.Trainer(accelerator="cuda")))
