import os
from plants_sm.data_standardization.truncation import Truncator
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert

def generate_features(transformer, transformer_name):

    datasets = ["test", "train", "validation"]

    for dataset_ in datasets:
        print(f"Generating features for {dataset_}")
        dataset = SingleInputDataset.from_csv(f"/scratch/jribeiro/ec_number_prediction/final_data/{dataset_}.csv", instances_ids_field="accession", representation_field="sequence", 
                            labels_field=slice(8, -1))
        
        Truncator(max_length=1545).fit_transform(dataset)
        transformer.fit_transform(dataset)

        os.makedirs(f"/scratch/jribeiro/ec_number_prediction/{transformer_name}", exist_ok=True)

        dataset.to_csv(f"/scratch/jribeiro/ec_number_prediction/{transformer_name}/{dataset_}_{transformer_name}.csv")

if __name__ == '__main__':
    transformer = ESMEncoder(device="cuda", esm_function="esm2_t36_3B_UR50D", batch_size=1, num_gpus=5)
    generate_features(transformer, "esm2_t36_3B_UR50D")

    transformer = ProtBert(device="cuda")
    generate_features(transformer, "protbert")