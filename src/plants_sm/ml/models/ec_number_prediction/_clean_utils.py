import random

import torch
import torch.nn.functional as F


def SupConHardLoss(model_emb, temp, n_pos):
    '''
    return the SupCon-Hard loss
    features:
        model output embedding, dimension [bsz, n_all, out_dim],
        where bsz is batchsize,
        n_all is anchor, pos, neg (n_all = 1 + n_pos + n_neg)
        and out_dim is embedding dimension
    temp:
        temperature
    n_pos:
        number of positive examples per anchor
    '''
    # l2 normalize every embedding
    features = F.normalize(model_emb, dim=-1, p=2)
    # features_T is [bsz, outdim, n_all], for performing batch dot product
    features_T = torch.transpose(features, 1, 2)
    # anchor is the first embedding
    anchor = features[:, 0]
    # anchor is the first embedding
    anchor_dot_features = torch.bmm(anchor.unsqueeze(1), features_T) / temp
    # anchor_dot_features now [bsz, n_all], contains
    anchor_dot_features = anchor_dot_features.squeeze(1)
    # deduct by max logits, which will be 1/temp since features are L2 normalized
    logits = anchor_dot_features - 1 / temp
    # the exp(z_i dot z_a) excludes the dot product between itself
    # exp_logits is of size [bsz, n_pos+n_neg]
    exp_logits = torch.exp(logits[:, 1:])
    exp_logits_sum = n_pos * torch.log(exp_logits.sum(1))  # size [bsz], scale by n_pos
    pos_logits_sum = logits[:, 1:n_pos + 1].sum(1)  # sum over all (anchor dot pos)
    log_prob = (pos_logits_sum - exp_logits_sum) / n_pos
    loss = - log_prob.mean()
    return loss



def mine_hard_negative(dist_map, knn=10):
    # print("The number of unique EC numbers: ", len(dist_map.keys()))
    ecs = list(dist_map.keys())
    negative = {}
    for i, target in enumerate(ecs):
        sort_orders = sorted(
            dist_map[target].items(), key=lambda x: x[1], reverse=False)
        if sort_orders[1][1] != 0:
            freq = [1 / i[1] for i in sort_orders[1:1 + knn]]
            neg_ecs = [i[0] for i in sort_orders[1:1 + knn]]
        elif sort_orders[2][1] != 0:
            freq = [1 / i[1] for i in sort_orders[2:2 + knn]]
            neg_ecs = [i[0] for i in sort_orders[2:2 + knn]]
        elif sort_orders[3][1] != 0:
            freq = [1 / i[1] for i in sort_orders[3:3 + knn]]
            neg_ecs = [i[0] for i in sort_orders[3:3 + knn]]
        else:
            freq = [1 / i[1] for i in sort_orders[4:4 + knn]]
            neg_ecs = [i[0] for i in sort_orders[4:4 + knn]]

        normalized_freq = [i / sum(freq) for i in freq]
        negative[target] = {
            'weights': normalized_freq,
            'negative': neg_ecs
        }
    return negative


def mine_negative(anchor, id_ec, ec_id, mine_neg):
    anchor_ec = id_ec[anchor]
    pos_ec = random.choice(anchor_ec)
    neg_ec = mine_neg[pos_ec]['negative']
    weights = mine_neg[pos_ec]['weights']
    result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    while result_ec in anchor_ec:
        result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    neg_id = random.choice(ec_id[result_ec])
    return neg_id


def random_positive(id, id_ec, ec_id):
    pos_ec = random.choice(id_ec[id])
    pos = id
    if len(ec_id[pos_ec]) == 1:
        return pos + '_' + str(random.randint(0, 9))
    while pos == id:
        pos = random.choice(ec_id[pos_ec])
    return pos


class MultiPosNegDatasetWithMineEC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg, n_pos, n_neg, train_dataset):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

        self.train_dataset = train_dataset

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        a = torch.from_numpy(self.train_dataset.features["place_holder"][anchor]).unsqueeze(0)
        data = [a]
        for _ in range(self.n_pos):
            pos = random_positive(anchor, self.id_ec, self.ec_id)
            p = torch.from_numpy(self.train_dataset.features["place_holder"][pos]).unsqueeze(0)
            data.append(p)
        for _ in range(self.n_neg):
            neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
            n = torch.from_numpy(self.train_dataset.features["place_holder"][neg]).unsqueeze(0)
            data.append(n)
        return torch.cat(data)


def get_dataloader(dist_map, id_ec, ec_id, n_pos, n_neg, train_dataset, batch_size):
    params = {
        'batch_size': batch_size,
        'shuffle': True,
    }
    negative = mine_hard_negative(dist_map, 100)
    train_data = MultiPosNegDatasetWithMineEC(
        id_ec, ec_id, negative, n_pos, n_neg, train_dataset)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader


