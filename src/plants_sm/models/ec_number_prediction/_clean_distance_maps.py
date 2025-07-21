import pickle
import re
import time

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

from plants_sm.data_structures.dataset import PLACEHOLDER_FIELD
from plants_sm.io.pickle import write_pickle


def get_ec_from_regex_match(match):
    if match is not None:
        EC = match.group()
        if EC is not None:
            return EC
    return None


def divide_labels_by_EC_level(ec_numbers):
    ECs = ec_numbers.split(";")
    # get the first 3 ECs with regular expression
    EC3 = []
    EC2 = []
    EC1 = []
    EC4 = []
    for EC in ECs:
        new_EC = re.search(r"^\d+.\d+.\d+.n*\d+", EC)
        new_EC = get_ec_from_regex_match(new_EC)
        if isinstance(new_EC, str):
            if new_EC not in EC4:
                EC4.append(new_EC)

        new_EC = re.search(r"^\d+.\d+.\d+", EC)
        new_EC = get_ec_from_regex_match(new_EC)
        if isinstance(new_EC, str):
            if new_EC not in EC3:
                EC3.append(new_EC)

        new_EC = re.search(r"^\d+.\d+", EC)
        new_EC = get_ec_from_regex_match(new_EC)
        if isinstance(new_EC, str):
            if new_EC not in EC2:
                EC2.append(new_EC)

        new_EC = re.search(r"^\d+", EC)
        new_EC = get_ec_from_regex_match(new_EC)
        if isinstance(new_EC, str):
            if new_EC not in EC1:
                EC1.append(new_EC)

    return EC1, EC2, EC3, EC4


def get_ec_id_dict(dataset, EC_label) -> tuple:
    id_ec = {}
    ec_id = {}
    labels_to_consider = dataset._labels_names

    for i, rows in dataset.dataframe.iterrows():
        sequence_id = rows[dataset.instances_ids_field]
        EC = rows[EC_label]
        if i > 0:
            EC1, EC2, EC3, EC4 = divide_labels_by_EC_level(EC)
            all_ecs = EC1 + EC2 + EC3 + EC4
            for ec in all_ecs:
                if ec not in labels_to_consider:
                    continue
                else:
                    if sequence_id not in id_ec.keys():
                        id_ec[sequence_id] = []
                        id_ec[sequence_id].append(ec)
                    else:
                        id_ec[sequence_id].append(ec)
            if ec in labels_to_consider:
                for ec in id_ec[sequence_id]:
                    if ec not in ec_id.keys():
                        ec_id[ec] = set()
                        ec_id[ec].add(sequence_id)
                    else:
                        ec_id[ec].add(sequence_id)
    return id_ec, ec_id


def esm_embedding(dataset, ec_id_dict, device, dtype):
    '''
    Loading esm embedding in the sequence of EC numbers
    prepare for calculating cluster center by EC
    '''
    esm_emb = []
    # for ec in tqdm(list(ec_id_dict.keys())):
    for ec in list(ec_id_dict.keys()):
        ids_for_query = list(ec_id_dict[ec])
        esm_to_cat = [torch.from_numpy(dataset.features[PLACEHOLDER_FIELD][id_]).unsqueeze(0) for id_ in ids_for_query]
        esm_emb = esm_emb + esm_to_cat
    return torch.cat(esm_emb).to(device=device, dtype=dtype)


def model_embedding_test(dataset, id_ec_test, model, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    then inferenced with model to get model embedding
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [torch.from_numpy(dataset.features[PLACEHOLDER_FIELD][id_]).unsqueeze(0)
                  for id_ in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    model_emb = model(esm_emb)
    return model_emb


def model_embedding_test_ensemble(dataset, id_ec_test, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [torch.from_numpy(dataset.features[PLACEHOLDER_FIELD][id_]).unsqueeze(0)
                  for id_ in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    return esm_emb


def get_dist_map_test(model_emb_train, model_emb_test,
                      ec_id_dict_train, id_ec_test,
                      device, dtype, dot=False):
    '''
    Get the pair-wise distance map for test queries and train EC cluster centers
    map is of size of (N_test_ids, N_EC_train)
    '''
    print("The embedding sizes for train and test:",
          model_emb_train.size(), model_emb_test.size())
    # get cluster center for all EC appeared in training set
    cluster_center_model = get_cluster_center(
        model_emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), model_emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    # calculate distance map between n_query_test * total_ec_n (training) pairs
    ids = list(id_ec_test.keys())
    print(f'Calculating eval distance map, between {len(ids)} test ids '
          f'and {total_ec_n} train EC cluster centers')
    if dot:
        eval_dist = dist_map_helper_dot(ids, model_emb_test, ecs, model_lookup)
    else:
        eval_dist = dist_map_helper(ids, model_emb_test, ecs, model_lookup)
    return eval_dist


def get_random_nk_dist_map(emb_train, rand_nk_emb_train,
                           ec_id_dict_train, rand_nk_ids,
                           device, dtype, dot=False):
    '''
    Get the pair-wise distance map between
    randomly chosen nk ids from training and all EC cluster centers
    map is of size of (nk, N_EC_train)
    '''

    cluster_center_model = get_cluster_center(emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    if dot:
        random_nk_dist_map = dist_map_helper_dot(
            rand_nk_ids, rand_nk_emb_train, ecs, model_lookup)
    else:
        random_nk_dist_map = dist_map_helper(
            rand_nk_ids, rand_nk_emb_train, ecs, model_lookup)
    return random_nk_dist_map


def random_nk_model(id_ec_train, ec_id_dict_train, emb_train, n=10, weighted=False):
    ids = list(id_ec_train.keys())
    nk = n * 1000
    if weighted:
        P = []
        for id in id_ec_train.keys():
            ecs_id = id_ec_train[id]
            ec_densities = [len(ec_id_dict_train[ec]) for ec in ecs_id]
            # the prob of calling this id is inversely prop to 1/max(density)
            P.append(1 / np.max(ec_densities))
        P = P / np.sum(P)
        random_nk_id = np.random.choice(
            range(len(ids)), nk, replace=True, p=P)
    else:
        random_nk_id = np.random.choice(range(len(ids)), nk, replace=False)

    random_nk_id = np.sort(random_nk_id)
    chosen_ids = [ids[i] for i in random_nk_id]
    chosen_emb_train = emb_train[random_nk_id]
    return chosen_ids, chosen_emb_train


def get_cluster_center(model_emb, ec_id_dict):
    cluster_center_model = {}
    id_counter = 0
    with torch.no_grad():
        for ec in tqdm(list(ec_id_dict.keys())):
            ids_for_query = list(ec_id_dict[ec])
            id_counter_prime = id_counter + len(ids_for_query)
            emb_cluster = model_emb[id_counter: id_counter_prime]
            cluster_center = emb_cluster.mean(dim=0)
            cluster_center_model[ec] = cluster_center.detach().cpu()
            id_counter = id_counter_prime
    return cluster_center_model


def dist_map_helper(keys1, lookup1, keys2, lookup2):
    dist = {}
    bar = tqdm(total=len(keys1))
    for i, key1 in enumerate(keys1):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        bar.update(1)
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist



def get_dist_map(ec_id_dict, esm_emb, device, dtype, model=None, dot=False):
    '''
    Get the distance map for training, size of (N_EC_train, N_EC_train)
    between all possible pairs of EC cluster centers
    '''
    # inference all queries at once to get model embedding
    if model is not None:
        model_emb = model(esm_emb.to(device=device, dtype=dtype))
    else:
        # the first distance map before training comes from ESM
        model_emb = esm_emb
    # calculate cluster center by averaging all embeddings in one EC
    cluster_center_model = get_cluster_center(model_emb, ec_id_dict)
    # organize cluster centers in a matrix
    total_ec_n, out_dim = len(ec_id_dict.keys()), model_emb.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    # calculate pairwise distance map between total_ec_n * total_ec_n pairs
    print(f'Calculating distance map, number of unique EC is {total_ec_n}')
    if dot:
        model_dist = dist_map_helper_dot(ecs, model_lookup, ecs, model_lookup)
    else:
        model_dist = dist_map_helper(ecs, model_lookup, ecs, model_lookup)
    return model_dist


def dist_map_helper_dot(keys1, lookup1, keys2, lookup2):
    dist = {}
    lookup1 = F.normalize(lookup1, dim=-1, p=2)
    lookup2 = F.normalize(lookup2, dim=-1, p=2)
    for i, key1 in tqdm(enumerate(keys1)):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm ** 2
        # dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist


def compute_esm_distance(dataset, distance_map_path, device):
    _, ec_id_dict = get_ec_id_dict(dataset, "EC")
    dtype = torch.float32
    esm_emb = esm_embedding(dataset, ec_id_dict, device, dtype)
    esm_dist = get_dist_map(ec_id_dict, esm_emb, device, dtype)
    write_pickle(distance_map_path + '.pkl', esm_dist)
    write_pickle(distance_map_path + '_esm.pkl', esm_emb)
