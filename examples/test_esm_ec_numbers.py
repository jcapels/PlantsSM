from plants_sm.data_standardization.truncation import Truncator
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder

def generate_esm_features():

    datasets = ["test", "train", "validation"]
    esms = ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"]
    
    for esm in esms:
        print(f"Generating features for {esm}")
        for dataset_ in datasets:
            print(f"Generating features for {dataset_}")
            dataset = SingleInputDataset.from_csv(f"./examples/data/{dataset_}.csv", instances_ids_field="accession", representation_field="sequence", 
                                labels_field=slice(8, -1))
            Truncator(max_length=1545).fit_transform(dataset)
            ESMEncoder(device="cuda", esm_function=esm, batch_size=1, num_gpus=4).fit_transform(dataset)
            dataset.to_csv(f"./examples/data/{dataset_}_{esm}.csv")



if __name__ == '__main__':

    generate_esm_features()


