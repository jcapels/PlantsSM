import os
import time

import numpy as np
import pandas as pd
from plants_sm.models.constants import FileConstants
import torch
from torch import nn
from tqdm import tqdm

from plants_sm.data_structures.dataset import Dataset
from plants_sm.io.pickle import read_pickle, write_pickle
from plants_sm.models.ec_number_prediction._clean_distance_maps import compute_esm_distance, get_dist_map, \
    get_ec_id_dict, model_embedding_test, get_dist_map_test, random_nk_model, get_random_nk_dist_map
from plants_sm.models.ec_number_prediction._clean_utils import SupConHardLoss, get_dataloader
from plants_sm.models.model import Model
from plants_sm.models.ec_number_prediction._clean_distance_maps import divide_labels_by_EC_level

from plants_sm.models._utils import write_model_parameters_to_pickle

import logging
import os
from logging.handlers import TimedRotatingFileHandler


class LayerNormNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class PValueInference:
    def __init__(self, p_value=1e-5, nk_random=20):
        self.p_value = p_value
        self.nk_random = nk_random


class CLEANSupConH(Model):

    def __init__(self,
                 distance_map_path: str, input_dim, hidden_dim=512, out_dim=256, dtype=torch.float32,
                 device: str = "cuda", drop_out=0.1,
                 lr=5e-4, epochs=1500, n_pos=9, n_neg=30, adaptative_rate=10,
                 temp=0.1, batch_size=6000, verbose=True, ec_label="EC",
                 model_name="CLEANSupConH", path_to_save_model="./data/model/",
                 evaluation_class=PValueInference(p_value=1e-5, nk_random=20), logger_path="clean_supconh.log"):
        self.distance_map_path = distance_map_path
        parent_folder = os.path.dirname(distance_map_path)
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)

        self.path_to_save_model = path_to_save_model
        if not os.path.exists(self.path_to_save_model):
            os.makedirs(self.path_to_save_model)

        self.device = device
        self.model = LayerNormNet(input_dim, hidden_dim, out_dim, device, dtype, drop_out)
        self.dtype = dtype
        self._history = {"train_loss": []}
        self.lr = lr
        self.epochs = epochs
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.adaptative_rate = adaptative_rate
        self.temp = temp
        self.batch_size = batch_size
        self.loss_function = SupConHardLoss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.verbose = verbose
        self.ec_label = ec_label
        self.model_name = model_name
        self.evaluation_class = evaluation_class

        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        if logger_path:
            handler = TimedRotatingFileHandler(logger_path, when='midnight', backupCount=30)
        else:
            handler = TimedRotatingFileHandler(f'./{self.model_name}.log', when='midnight', backupCount=20)
        handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

        self.rand_nk_ids = None
        self.rand_nk_emb_train = None
        self.random_nk_dist_map = None
    
    def __name__(self):
        return "CLEANSupConH"

    def _preprocess_data(self, dataset: Dataset, **kwargs) -> Dataset:
        pass

    def _fit_data(self, train_dataset: Dataset, validation_dataset: Dataset):
        self.labels = train_dataset._labels_names
        if not os.path.exists(self.distance_map_path + ".pkl") \
                and not os.path.exists(self.distance_map_path + "_esm.pkl"):
            compute_esm_distance(train_dataset, self.distance_map_path, self.device)

        best_loss = float('inf')
        # ======================== override args ====================#
        self.logger.info(f'==> device used: {self.device} | dtype used: {self.dtype}\n==> args:')
        # ======================== ESM embedding  ===================#
        # loading ESM embedding for dist map

        self.logger.info(f'==> loading ESM embedding for dist map')
        self._esm_emb_train = read_pickle(self.distance_map_path + '_esm.pkl').to(device=self.device, dtype=self.dtype)
        dist_map = read_pickle(self.distance_map_path + '.pkl')

        self._id_ec_train, self._ec_id_dict_train = get_ec_id_dict(train_dataset, self.ec_label)
        ec_id = {key: list(self._ec_id_dict_train[key]) for key in self._ec_id_dict_train.keys()}
        # ======================== initialize model =================#
        train_loader = get_dataloader(dist_map, self._id_ec_train, ec_id, self.n_pos, self.n_neg, train_dataset,
                                      self.batch_size)
        
        self.logger.info(f"The number of unique EC numbers: {len(dist_map.keys())}")

        # ======================== training =======-=================#
        # training
        for epoch in range(1, self.epochs + 1):
            if epoch % self.adaptative_rate == 0 and epoch != self.epochs + 1:
                # save updated model
                torch.save(self.model.state_dict(), os.path.join(self.path_to_save_model,
                                                                 self.model_name + '_' + str(epoch) + '.pth'))
                # delete last model checkpoint
                if epoch != self.adaptative_rate:
                    os.remove(os.path.join(self.path_to_save_model, self.model_name + '_' +
                                           str(epoch - self.adaptative_rate) + '.pth'))
                # sample new distance map
                dist_map = get_dist_map(
                    self._ec_id_dict_train, self._esm_emb_train, self.device, self.dtype, model=self.model)
                train_loader = get_dataloader(dist_map, self._id_ec_train, ec_id, self.n_pos, self.n_neg, train_dataset,
                                              self.batch_size)
            # -------------------------------------------------------------------- #
            epoch_start_time = time.time()
            train_loss = self._train(train_loader, epoch)
            # only save the current best model near the end of training
            if train_loss < best_loss and epoch > 0.8 * self.epochs:
                torch.save(self.model.state_dict(), os.path.join(self.path_to_save_model, self.model_name + '.pth'))
                best_loss = train_loss
                self.logger.info(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')

            elapsed = time.time() - epoch_start_time
            self.logger.info('-' * 75)
            self.logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'training loss {train_loss:6.4f}')
            self.logger.info('-' * 75)
            if "loss" not in self._history:
                self._history["loss"] = []
            self._history["loss"].append(train_loss)

            # emb_train = self.model(self._esm_emb_train)

            # self.rand_nk_ids, self.rand_nk_emb_train = random_nk_model(
            #     self._id_ec_train, self._ec_id_dict_train, emb_train, n=self.evaluation_class.nk_random, weighted=True)

            # self.random_nk_dist_map = get_random_nk_dist_map(
            #     emb_train, self.rand_nk_emb_train, self._ec_id_dict_train, self.rand_nk_ids, self.device, self.dtype)
            
            # predictions = self._predict(validation_dataset)
            # self._history["validation_loss"] = self.evaluation_class.evaluate(validation_dataset, predictions)
            
        # remove tmp save weights
        if os.path.exists(os.path.join(self.path_to_save_model, self.model_name + '.pth')):
            os.remove(os.path.join(self.path_to_save_model, self.model_name + '.pth'))
        if os.path.exists(os.path.join(self.path_to_save_model, self.model_name + '_' + str(epoch) + '.pth')):
            os.remove(os.path.join(self.path_to_save_model, self.model_name + '_' + str(epoch) + '.pth'))
        # save final weights
        torch.save(self.model.state_dict(), os.path.join(self.path_to_save_model, self.model_name + '.pth'))

    def _train(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.
        start_time = time.time()

        for batch, data in enumerate(train_loader):
            self.optimizer.zero_grad()
            model_emb = self.model(data.to(device=self.device, dtype=self.dtype))
            loss = self.loss_function(model_emb, self.temp, self.n_pos)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if self.verbose:
                ms_per_batch = (time.time() - start_time) * 1000
                cur_loss = total_loss
                self.logger.info(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                      f'lr {self.lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                      f'loss {cur_loss:5.2f}')
                start_time = time.time()
            self.logger.info(f"batch {batch} done")
        # record running average training loss
        return total_loss / (batch + 1)

    def _predict_proba(self, dataset: Dataset) -> np.ndarray:
        return self._predict(dataset)

    @staticmethod
    def _get_pvalue_choices(df, random_nk_dist_map, p_value=1e-5):
        all_test_EC = set()
        nk = len(random_nk_dist_map.keys())
        threshold = p_value * nk
        ecs = []
        for col in tqdm(df.columns):
            ec = []
            smallest_10_dist_df = df[col].nsmallest(10)
            for i in range(10):
                EC_i = smallest_10_dist_df.index[i]
                # find all the distances in the random nk w.r.t. EC_i
                # then sorted the nk distances
                rand_nk_dists = [random_nk_dist_map[rand_nk_id][EC_i]
                                 for rand_nk_id in random_nk_dist_map.keys()]
                rand_nk_dists = np.sort(rand_nk_dists)
                # rank dist_i among rand_nk_dists
                dist_i = smallest_10_dist_df[i]
                rank = np.searchsorted(rand_nk_dists, dist_i)
                if (rank <= threshold) or (i == 0):
                    dist_str = "{:.4f}".format(dist_i)
                    all_test_EC.add(EC_i)
                    ec.append('EC:' + str(EC_i) + '/' + dist_str)
                else:
                    break
            ec.insert(0, col)
            ecs.append(ec)
        return ecs
    
    def get_final_labels(self, EC1, EC2, EC3, EC4):
        res = np.zeros((len(self.labels),), dtype=object)
        
        for ec in EC1 + EC2 + EC3 + EC4:
            if ec in self.labels:
                index_ = self.labels.index(ec)
                res[index_] = 1

        return res

    def _predict(self, dataset: Dataset) -> np.ndarray:
        id_ec_test, _ = get_ec_id_dict(dataset, self.ec_label)
        # load precomputed EC cluster center embeddings if possible
        emb_train = self.model(self._esm_emb_train)

        if self.rand_nk_ids is None or self.rand_nk_emb_train is None or self.random_nk_dist_map is None:
            self.rand_nk_ids, self.rand_nk_emb_train = random_nk_model(
                self._id_ec_train, self._ec_id_dict_train, emb_train, n=self.evaluation_class.nk_random, weighted=True)

            self.random_nk_dist_map = get_random_nk_dist_map(
                emb_train, self.rand_nk_emb_train, self._ec_id_dict_train, self.rand_nk_ids, self.device, self.dtype)

        emb_test = model_embedding_test(dataset, id_ec_test, self.model, self.device, self.dtype)

        eval_dist = get_dist_map_test(emb_train, emb_test, self._ec_id_dict_train, id_ec_test,
                                      self.device, self.dtype)

        eval_df = pd.DataFrame.from_dict(eval_dist)

        ecs = self._get_pvalue_choices(eval_df, self.random_nk_dist_map, p_value=self.evaluation_class.p_value)
        
        pred_label = []
        for row in ecs:
            preds_ec_lst = []
            preds_with_dist = row[1:]
            for pred_ec_dist in preds_with_dist:
                # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
                ec_i = pred_ec_dist.split(":")[1].split("/")[0]
                preds_ec_lst.append(ec_i)
            pred_label.append(";".join(preds_ec_lst))

        res = np.zeros((dataset.dataframe.shape[0], len(self.labels)), )
        for i, ec in enumerate(pred_label):
            
            EC1, EC2, EC3, EC4 = divide_labels_by_EC_level(ec)
            res[i, :] = self.get_final_labels(EC1, EC2, EC3, EC4)

        return res

    @staticmethod
    def _save_pytorch_model(model, path):
        weights_path = os.path.join(path, FileConstants.PYTORCH_MODEL_WEIGHTS.value)
        torch.save(model.state_dict(), weights_path)
        write_pickle(os.path.join(path, FileConstants.PYTORCH_MODEL_PKL.value), model)


    def _save(self, path: str):
        self._save_pytorch_model(self.model, path)

        del self.rand_nk_ids
        del self.rand_nk_emb_train
        del self.random_nk_dist_map

        self.rand_nk_ids = None
        self.rand_nk_emb_train = None
        self.random_nk_dist_map = None

        write_pickle(os.path.join(path, "model_class.pkl"), self)

    @staticmethod
    def _read_model(path):
        """
        Read the model from the specified path.

        Parameters
        ----------
        path: str
            Path to read the model from

        Returns
        -------
        torch.nn.Module
        """
        weights_path = os.path.join(path, FileConstants.PYTORCH_MODEL_WEIGHTS.value)
        model = read_pickle(os.path.join(path, FileConstants.PYTORCH_MODEL_PKL.value))
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        return model

    @classmethod
    def _load(cls, path: str):
        new_class = read_pickle(os.path.join(path, "model_class.pkl"))
        model = cls._read_model(path)
        new_class.model = model

        return new_class

    @property
    def history(self):
        return self._history
