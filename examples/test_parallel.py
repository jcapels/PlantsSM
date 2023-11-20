

from joblib import Parallel, delayed
import torch
from tqdm import tqdm
from plants_sm.data_structures.dataset.dataset import Dataset
from plants_sm.data_structures.dataset.single_input_dataset import PLACEHOLDER_FIELD, SingleInputDataset


from plants_sm.data_standardization.truncation import Truncator
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_FUNCTIONS, ESM_LAYERS

from plants_sm.models.ec_number_prediction.esm import EC_ESM1b_Lightning, EC_ESM_Lightning
from plants_sm.models._esm import ESM

from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import EarlyStopping, BaseFinetuning

from sklearn.metrics import f1_score

import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def _preprocess_data(dataset: Dataset, model_name) -> DataLoader:
    tensors = []
    sequences = [(sequence_id, dataset.instances[PLACEHOLDER_FIELD][sequence_id]) for sequence_id in
                    dataset.dataframe[dataset.instances_ids_field]]
    
    esm_callable = ESM_FUNCTIONS[model_name]

    _, alphabet = esm_callable()
    batch_converter = alphabet.get_batch_converter()

    # _, _, tokens = batch_converter(sequences)

    batch_size = 10000  # You can adjust this based on your preferences

    # Initialize the progress bar
    progress_bar = tqdm(total=len(sequences), desc="Processing sequences", position=0, leave=True)

    # Define the function to be parallelized
    def process_batch(batch):
        _, _, tokens = batch_converter(batch)
        return tokens

    # Process sequences in parallel with a progress bar in batches
    result_tokens = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_results = Parallel(n_jobs=-1)(delayed(process_batch)(batch) for batch in [batch])
        result_tokens.extend(batch_results)
        progress_bar.update(len(batch))

    # Close the progress bar
    progress_bar.close()

    # Use joblib to parallelize the function across sequences
    result_tokens = torch.cat(result_tokens, dim=0)
    tensors.append(result_tokens)

    try:
        if dataset.y is not None:
            tensors.append(torch.tensor(dataset.y, dtype=torch.float))
    except ValueError:
        pass

    dataset = TensorDataset(
        *tensors
    )
    return dataset


def f1_score_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

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

    # train_dataset = _preprocess_data(train_dataset, "esm2_t6_8M_UR50D")
    # torch.save(train_dataset, '/home/jcapela/PlantsSM/examples/tensor_train_dataset.pt')
    # validation_dataset = _preprocess_data(validation_dataset, "esm2_t6_8M_UR50D")
    # torch.save(validation_dataset, '/home/jcapela/PlantsSM/examples/tensor_validation_dataset.pt')

    sharding_strategy="FULL_SHARD"
    limit_all_gathers=True
    cpu_offload=True
    callbacks = EarlyStopping("val_metric", patience=5, mode="max")

    strategy = FSDPStrategy(sharding_strategy=sharding_strategy, limit_all_gathers=limit_all_gathers, cpu_offload=cpu_offload)
    

    tensor_train_dataset = torch.load('/home/jcapela/PlantsSM/examples/tensor_train_dataset.pt', map_location=torch.device('cpu'))
    tensor_validation_dataset = torch.load('/home/jcapela/PlantsSM/examples/tensor_validation_dataset.pt', map_location=torch.device('cpu'))

    tensor_train_dataset = DataLoader(
            tensor_train_dataset,
            shuffle=True,
            batch_size=6
        )
    tensor_validation_dataset = DataLoader(
            tensor_validation_dataset,
            shuffle=False,
            batch_size=6
        )

    # model = EC_ESM_Lightning("esm2_t12_35M_UR50D", [2560, 5120], 2771, batch_size=2)
    # layers_1 = ESM_LAYERS["esm2_t12_35M_UR50D"] - 1
    # layer_2 = ESM_LAYERS["esm2_t12_35M_UR50D"] - 2
    # no_grad = set()
    # for parameter in model.named_parameters():
    #     if "layers" in parameter[0] and str(layers_1) not in parameter[0] and str(layer_2) not in parameter[0]:
    #         no_grad.add(parameter[0])

    model = EC_ESM_Lightning("esm2_t6_8M_UR50D",[2560, 5120], 2771, metric=f1_score_macro)
    model_ = ESM(module=model, 
                    max_epochs=30,
                    batch_size=1,
                    devices=[1,2, 4, 5, 6, 7],
                    accelerator="gpu",
                    strategy=strategy,
                    callbacks=[callbacks])
    model_.fit(tensor_train_dataset, validation_dataset=tensor_validation_dataset)
    
    # model.fit(train_dataset, validation_dataset=validation_dataset)
    # model.save("test_model")

    # model = EC_ESM_Lightning.load_from_checkpoint("/home/jcapela/PlantsSM/examples/test_model/pytorch_model_weights.ckpt",
    #                                             model_name="esm2_t12_35M_UR50D", hidden_layers=[2560, 5120], num_classes=2771, 
    #                                             batch_size=16)
    # model = EC_ESM_Lightning.load("/home/jcapela/PlantsSM/examples/test_model/")
    # print(model.predict(dataset, trainer = L.Trainer(accelerator="cuda")))

