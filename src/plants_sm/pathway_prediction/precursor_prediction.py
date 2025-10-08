import os

import numpy as np

import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from deepmol.pipeline import Pipeline
from deepmol.datasets import SmilesDataset

labels_ = {
        'C00341': 'Geranyl diphosphate',
        'C01789': 'Campesterol',
        'C00078': 'Tryptophan',
        'C00049': 'L-Aspartate',
        'C00183': 'L-Valine',
        'C03506': 'Indoleglycerol phosphate',
        'C00187': 'Cholesterol',
        'C00079': 'L-Phenylalanine',
        'C00047': 'L-Lysine',
        'C01852': 'Secologanin',
        'C00407': 'L-Isoleucine',
        'C00129': 'Isopentenyl diphosphate',
        'C00235': 'Dimethylallyl diphosphate',
        'C00062': 'L-Arginine',
        'C00353': 'Geranylgeranyl diphosphate',
        'C00148': 'L-Proline',
        'C00073': 'L-Methionine',
        'C00108': 'Anthranilate',
        'C00123': 'L-Leucine',
        'C00135': 'L-Histidine',
        'C00448': 'Farnesyl diphosphate',
        'C00082': 'L-Tyrosine',
        'C00041': 'L-Alanine',
        'C00540': 'Cinnamoyl-CoA',
        'C01477': 'Apigenin',
        'C05903': 'Kaempferol',
        'C05904': 'Pelargonin',
        'C05905': 'Cyanidin',
        'C05908': 'Delphinidin',
        'C00389': 'Quercetin',
        'C01514': 'Luteolin',
        'C09762': "Liquiritigenin",
        'C00509': 'Naringenin',
        'C00223': 'p-Coumaroyl-CoA'
    }

smiles = {
    
}

kegg_labels = ['C00073', 'C00078', 'C00079', 'C00082', 'C00235', 'C00341', 'C00353',
              'C00448', 'C01789', 'C03506', 'C00047', 'C00108', 'C00187', 'C00148',
              'C00041', 'C00129', 'C00062', 'C01852', 'C00049', 'C00135', 'C00223',
              'C00509', 'C00540', 'C01477', 'C05903', 'C05904', 'C05905', 'C05908',
              'C09762']


def convert_predictions_into_model_names(predictions):

    labels_names = np.array([labels_[label] for label in kegg_labels])
    ones = predictions == 1
    labels_all = []
    for i, prediction in enumerate(ones):
        labels_all.append(";".join(labels_names[prediction]))
    return labels_all

def export_precursors(smiles: list):
    try:
        # read text file
        dataset = SmilesDataset(smiles=smiles)

        predictions = predict_from_dataset(dataset)

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        building_blocks_list_path = os.path.join(BASE_DIR,
                                        "pathway_prediction",
                                        "precursor_prediction",
                                        "building_blocks_to_smiles.txt")
        
        building_blocks_output_path = os.path.join(BASE_DIR,
                                        "pathway_prediction",
                                        "precursor_prediction",
                                        "building_blocks.txt")

        df = pd.read_csv(building_blocks_list_path, header=None, sep='\t')
        df.columns = ['precursor',"SMILES"]

        unique_precursors = set()
        for precursor_list in predictions:
            if precursor_list:
                unique_precursors.update(precursor_list.split(';'))

        df = df[df['precursor'].isin(unique_precursors)]
        if df.empty:
            print("No matching precursors found.")
            return False
    
        df.to_csv(building_blocks_output_path, index=False, sep='\t', header=False)

    except Exception as e:
        print(f"Error occurred while saving building blocks: {e}")
        return False

    return True

def predict_from_dataset(dataset):
    

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pretrain_model_path = os.path.join(BASE_DIR,
                                       "pathway_prediction",
                                       "precursor_prediction",
                                       "ridge")
    
    best_pipeline = Pipeline.load(pretrain_model_path)

    if dataset.mols.shape[0] == 0:
        raise ValueError("No molecules found in the dataset. The one provided is not valid.")
    predictions = best_pipeline.predict(dataset)
    predictions = convert_predictions_into_model_names(predictions)

    return predictions


def predict_precursors(smiles: list):

    dataset = SmilesDataset(smiles=smiles)
    return predict_from_dataset(dataset)

