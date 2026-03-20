import datetime
import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from hurry.filesize import size
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset

from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_standardization.truncation import Truncator
from plants_sm.pathway_prediction.esi_annotator import ESM2ESIAnnotator, ESM1bESIAnnotator, ProtBertESIAnnotator



def predict_esi(dataset):
    
    annotator = ESM2ESIAnnotator(num_gpus=4, device="cuda:0").annotate_from_file(dataset, "csv")

def predict_esi_esm1b(dataset):
    
    annotator = ESM1bESIAnnotator(device="cuda").annotate_from_file(dataset, "csv")


def predict_esi_protbert(dataset):
    
    annotator = ProtBertESIAnnotator(device="cuda").annotate_from_file(dataset, "csv")

def benchmark_resources_inference():
    datasets = [
        "dataset_100_100.csv", "dataset_300_1000.csv", "dataset_700_5000.csv", 
                "dataset_7000.csv", 
                "curated_dataset.csv"]
    
    if os.path.exists("benchmark_results.csv"):
        results = pd.read_csv("benchmark_results.csv")
    else:
        results = pd.DataFrame()

    for dataset in datasets:
        dataset_df = pd.read_csv(dataset)
        tracemalloc.start()
        start = time.time()

        predict_esi_protbert(dataset)

        end = time.time()
        print("Time spent: ", end - start)
        print("Memory needed: ", tracemalloc.get_traced_memory())
        unique_substrates_dataset = np.unique(dataset_df["Substrate ID"])
        num_unique_substrates = len(unique_substrates_dataset)
        unique_enzymes_dataset = np.unique(dataset_df["Enzyme ID"])
        num_unique_enzymes = len(unique_enzymes_dataset)
        num_rows = dataset_df.shape[0]

        results = pd.concat((results, 
                                pd.DataFrame({
                                                "unique_enzymes": [num_unique_enzymes],
                                                "unique_substrates": [num_unique_substrates],
                                                "num_pairs": [num_rows],
                                                "time": [str(datetime.timedelta(seconds=end - start))], 
                                                "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                ignore_index=True, axis=0)
        tracemalloc.stop()

        results.to_csv("benchmark_results.csv", index=False)



if __name__=="__main__":
    benchmark_resources_inference()