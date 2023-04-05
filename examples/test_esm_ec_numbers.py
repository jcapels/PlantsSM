from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder


if __name__ == '__main__':
    dataset = SingleInputDataset.from_csv("./examples/data/test_head.csv", instances_ids_field="accession", representation_field="sequence", 
                                labels_field=slice(8, -1))

    ESMEncoder(device="cuda", esm_function="esm2_t6_8M_UR50D", batch_size=2, num_gpus=3).fit_transform(dataset)

    print(dataset.X.shape)



