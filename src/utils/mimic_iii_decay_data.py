import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset

class MIMICDecayData:
    def __init__(self, batch_size, window_size, file_path, train_frac=0.7, dev_frac= 0.15, test_frac=0.15):
        self.batch_size = batch_size
        self.window_size = window_size
        self.train_frac = train_frac
        self.dev_frac = dev_frac
        self.test_frac = test_frac

        all_data = np.load(file_path, allow_pickle=True)

        self.x = all_data['x']
        self.y = all_data['y']
        self.statics = all_data['statics']
        self.mask = all_data['mask']
        self.delta = all_data['delta']
        self.x_mean = all_data['xmean']
        self.last_observed = all_data['lastob']

        self.statics_size = self.statics.shape[-1]
        self.input_size  = self.x.shape[-1]
        # self.test_instances = 0

        del all_data
    
    def data_loader(self):
        self.x = np.reshape(self.x, (self.y.shape[0], self.window_size, -1))
        self.y = np.squeeze(self.y, axis=1)
        
        self.mask = np.reshape(self.mask, (self.y.shape[0], self.window_size, -1))
        self.delta = np.reshape(self.delta, (self.y.shape[0], self.window_size, -1))
        self.last_observed = np.reshape(self.last_observed, (self.y.shape[0], self.window_size, -1))

        index_ = np.arange(self.x.shape[0], dtype = int)
        np.random.seed(1024)
        np.random.shuffle(index_)
        
        self.x = self.x[index_]
        self.y = self.y[index_]
        self.statics = self.statics[index_]
        self.mask = self.mask[index_]
        self.delta = self.delta[index_]
        self.last_observed = self.last_observed[index_]
        self.x_mean = self.x_mean[index_]

        # expanding dims to concatenate
        self.x = np.expand_dims(self.x, axis=1)
        self.mask = np.expand_dims(self.mask, axis=1)
        self.delta = np.expand_dims(self.delta, axis=1)
        self.last_observed = np.expand_dims(self.last_observed, axis=1)

        self.data_agg = np.concatenate((self.x, self.mask, self.delta, self.last_observed), axis=1)

        self.train_instances = int(self.x.shape[0] *  self.train_frac)
        self.dev_instances = int(self.x.shape[0] * self.dev_frac)
        self.test_instances = int(self.x.shape[0] * self.test_frac)

        self.train_data, self.train_label = self.data_agg[:self.train_instances], self.y[:self.train_instances]
        self.valid_data, self.valid_label = self.data_agg[self.train_instances:self.train_instances + self.dev_instances], self.y[self.train_instances:self.train_instances + self.dev_instances]
        self.test_data, self.test_label = self.data_agg[-self.test_instances:], self.y[-self.test_instances:]

        self.train_static= self.statics[:self.train_instances]
        self.valid_static = self.statics[self.train_instances: self.train_instances +self.dev_instances]
        self.test_static = self.statics[-self.test_instances:]

        self.train_x_mean = self.x_mean[:self.train_instances]
        self.valid_x_mean = self.x_mean[self.train_instances: self.train_instances +self.dev_instances]
        self.test_x_mean = self.x_mean[-self.test_instances:]

        self.train_data, self.train_label = torch.from_numpy(self.train_data), torch.from_numpy(self.train_label)
        self.valid_data, self.valid_label = torch.from_numpy(self.valid_data), torch.from_numpy(self.valid_label)
        self.test_data, self.test_label =torch.from_numpy(self.test_data), torch.from_numpy(self.test_label)

        self.train_static = torch.from_numpy(self.train_static)
        self.valid_static = torch.from_numpy(self.valid_static)
        self.test_static = torch.from_numpy(self.test_static)

        self.train_x_mean = torch.from_numpy(self.train_x_mean)
        self.test_x_mean = torch.from_numpy(self.test_x_mean)
        self.valid_x_mean = torch.from_numpy(self.valid_x_mean)

        train_dataset = TensorDataset(self.train_data, self.train_static, self.train_x_mean, self.train_label)
        val_dataset = TensorDataset(self.valid_data, self.valid_static, self.valid_x_mean, self.valid_label)
        test_dataset = TensorDataset(self.test_data, self.test_static, self.test_x_mean, self.test_label)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True)

        return train_loader, val_loader, test_loader


# if __name__ =='__main__':
#     d = MIMICDecayData(64, 24 , './data/mimic_iii/test_dump/decay_data_20926.npz')
#     t, v, ts = d.data_loader()
#     print(len(t))
#     for i, x,s, m,y in tqdm(ran(t)):
#         print(f'x:{x.size()},s:{s.size()}, m:{m.size()}, y:{y.size()}')