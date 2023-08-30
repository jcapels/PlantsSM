import os
import time

import numpy as np
import torch
from torch import nn

from plants_sm.data_structures.dataset import Dataset
from plants_sm.io.pickle import read_pickle
from plants_sm.models.ec_number_prediction._clean_distance_maps import compute_esm_distance, get_dist_map, \
    get_ec_id_dict
from plants_sm.models.ec_number_prediction._clean_utils import SupConHardLoss, get_dataloader
from plants_sm.models.model import Model


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


class CLEANSupConH(Model):

    def __init__(self,
                 distance_map_path: str, input_dim, hidden_dim=512, out_dim=256, dtype=torch.float32,
                 device: str = "cuda", drop_out=0.1,
                 lr=5e-4, epochs=1500, n_pos=9, n_neg=30, adaptative_rate=10,
                 temp=0.1, batch_size=6000, verbose=True, ec_label="EC",
                 model_name="CLEANSupConH", path_to_save_model="./data/model/"):
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
        self._history = {"train_loss": [], "val_loss": []}
        self.lr = lr
        self.epochs = epochs
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.adaptative_rate = adaptative_rate
        self.temp = temp
        self.batch_size = batch_size
        self.criterion = SupConHardLoss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.verbose = verbose
        self.ec_label = ec_label
        self.model_name = model_name

    def _preprocess_data(self, dataset: Dataset, **kwargs) -> Dataset:
        pass

    def _fit_data(self, train_dataset: Dataset, validation_dataset: Dataset):
        if not os.path.exists(self.distance_map_path + ".pkl") \
                and not os.path.exists(self.distance_map_path + "_esm.pkl"):
            compute_esm_distance(train_dataset, self.distance_map_path, self.device)

        best_loss = float('inf')
        # ======================== override args ====================#
        print('==> device used:', self.device, '| dtype used: ',
              self.dtype, "\n==> args:")
        # ======================== ESM embedding  ===================#
        # loading ESM embedding for dist map

        esm_emb = read_pickle(self.distance_map_path + 'esm.pkl').to(device=self.device, dtype=self.dtype)
        dist_map = read_pickle(self.distance_map_path + '.pkl')

        id_ec, ec_id_dict = get_ec_id_dict(train_dataset, "EC")
        ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}
        # ======================== initialize model =================#
        train_loader = get_dataloader(dist_map, id_ec, ec_id, self.n_pos, self.n_neg, train_dataset)
        print("The number of unique EC numbers: ", len(dist_map.keys()))
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
                    ec_id_dict, esm_emb, self.device, self.dtype, model=self.model)
                train_loader = get_dataloader(dist_map, id_ec, ec_id, self.n_pos, self.n_neg, train_dataset)
            # -------------------------------------------------------------------- #
            epoch_start_time = time.time()
            train_loss = self._train(train_loader, epoch)
            # only save the current best model near the end of training
            if train_loss < best_loss and epoch > 0.8 * self.epochs:
                torch.save(self.model.state_dict(), os.path.join(self.path_to_save_model, self.model_name + '.pth'))
                best_loss = train_loss
                print(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')

            elapsed = time.time() - epoch_start_time
            print('-' * 75)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'training loss {train_loss:6.4f}')
            print('-' * 75)
        # remove tmp save weights
        os.remove(os.path.join(self.path_to_save_model, self.model_name + '.pth'))
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
            loss = self.criterion(model_emb, self.temp, self.n_pos)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if self.verbose:
                ms_per_batch = (time.time() - start_time) * 1000
                cur_loss = total_loss
                print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                      f'lr {self.lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                      f'loss {cur_loss:5.2f}')
                start_time = time.time()
        # record running average training loss
        return total_loss / (batch + 1)

    def _predict_proba(self, dataset: Dataset) -> np.ndarray:
        pass

    def _predict(self, dataset: Dataset) -> np.ndarray:
        pass

    def _save(self, path: str):
        pass

    @classmethod
    def _load(cls, path: str):
        pass

    @property
    def history(self):
        pass
