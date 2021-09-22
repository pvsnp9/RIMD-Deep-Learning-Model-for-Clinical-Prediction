import math
from time import process_time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class MIMICDataLoader:
    def __init__(self, window_size, file_path, train_frac=0.7, dev_frac=0.15, test_frac=0.15,
                 balance=False):
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
        self.input_size = self.x.shape[-1]

        self.x = np.reshape(self.x, (self.y.shape[0], self.window_size, -1))
        self.y = np.squeeze(self.y, axis=1)

        self.mask = np.reshape(self.mask, (self.y.shape[0], self.window_size, -1))
        self.delta = np.reshape(self.delta, (self.y.shape[0], self.window_size, -1))
        self.last_observed = np.reshape(self.last_observed, (self.y.shape[0], self.window_size, -1))

        # delta normalization for each patients
        delta_mean = np.amax(self.delta, axis=1)
        for i in range(self.delta.shape[0]):
            self.delta[i] = np.divide(self.delta[i], delta_mean[i], out=np.zeros_like(self.delta[i]),
                                      where=delta_mean[i] != 0)

        del all_data, delta_mean
        print('Seeding and creating shuffle')
        self.index_ = np.arange(self.x.shape[0], dtype=int)
        np.random.shuffle(self.index_)

    def prepare_data_loader(self, model, batch_size):
        self.batch_size = batch_size
        if model.startswith('RIMD') or model.startswith('GRUD'):
            print('Preparing Decay data')
            self.prepare_decay_loader()
        else:
            print('Preparing regular data')
            self.prepare_normal_loader()

        print('Done ')

    def prepare_normal_loader(self):
        x = self.x[self.index_]
        y = self.y[self.index_]
        statics = self.statics[self.index_]

        train_instances = int(x.shape[0] * self.train_frac)
        dev_instances = int(self.x.shape[0] * self.dev_frac)
        test_instances = int(self.x.shape[0] * self.test_frac)
        print(f'#### Test instances for data #{test_instances} with batch {self.batch_size}')

        train_x = x[:train_instances]
        dev_x = x[train_instances:train_instances + dev_instances]
        test_x = x[-test_instances:]

        train_ys = y[:train_instances]
        dev_ys = y[train_instances:train_instances + dev_instances]
        test_ys = y[-test_instances:]

        train_statics = statics[:train_instances]
        dev_statics = statics[train_instances:train_instances + dev_instances]
        test_statics = statics[-test_instances:]

        train_data, train_label = torch.from_numpy(train_x), torch.from_numpy(train_ys)
        val_data, val_label = torch.from_numpy(dev_x), torch.from_numpy(dev_ys)
        test_data, test_label = torch.from_numpy(test_x), torch.from_numpy(test_ys)

        train_static, val_static = torch.from_numpy(train_statics), torch.from_numpy(dev_statics)
        test_static = torch.from_numpy(test_statics)

        train_dataset = TensorDataset(train_data, train_static, train_label)
        val_dataset = TensorDataset(val_data, val_static, val_label)
        test_dataset = TensorDataset(test_data, test_static, test_label)

        self.normal_train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.normal_val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True )
        self.normal_test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True )

    def prepare_decay_loader(self):
        x = self.x[self.index_]
        y = self.y[self.index_]
        statics = self.statics[self.index_]
        mask = self.mask[self.index_]
        delta = self.delta[self.index_]
        last_observed = self.last_observed[self.index_]
        x_mean = self.x_mean[self.index_]

        # expanding dims to concatenate
        x = np.expand_dims(x, axis=1)
        mask = np.expand_dims(mask, axis=1)
        delta = np.expand_dims(delta, axis=1)
        last_observed = np.expand_dims(last_observed, axis=1)

        data_agg = np.concatenate((x, mask, delta, last_observed), axis=1)

        train_instances = int(x.shape[0] * self.train_frac)
        dev_instances = int(x.shape[0] * self.dev_frac)
        test_instances = int(x.shape[0] * self.test_frac)
        print(f'#### Test instances for decay data #{test_instances} with batch {self.batch_size}')

        train_data, train_label = data_agg[:train_instances], y[:train_instances]
        valid_data, valid_label = data_agg[train_instances:train_instances + dev_instances], y[
                                                                                             train_instances:train_instances + dev_instances]
        test_data, test_label = data_agg[-test_instances:], y[-test_instances:]

        train_static = statics[:train_instances]
        valid_static = statics[train_instances: train_instances + dev_instances]
        test_static = statics[-test_instances:]

        train_x_mean = x_mean[:train_instances]
        valid_x_mean = x_mean[train_instances: train_instances + dev_instances]
        test_x_mean = x_mean[-test_instances:]

        train_data, train_label = torch.from_numpy(train_data), torch.from_numpy(train_label)

        valid_data, valid_label = torch.from_numpy(valid_data), torch.from_numpy(valid_label)
        test_data, test_label = torch.from_numpy(test_data), torch.from_numpy(test_label)

        train_static = torch.from_numpy(train_static)
        valid_static = torch.from_numpy(valid_static)
        test_static = torch.from_numpy(test_static)

        train_x_mean = torch.from_numpy(train_x_mean)
        test_x_mean = torch.from_numpy(test_x_mean)
        valid_x_mean = torch.from_numpy(valid_x_mean)

        train_dataset = TensorDataset(train_data, train_static, train_x_mean, train_label)
        val_dataset = TensorDataset(valid_data, valid_static, valid_x_mean, valid_label)
        test_dataset = TensorDataset(test_data, test_static, test_x_mean, test_label)

        self.decay_train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.decay_val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        self.decay_test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

    def normal_data_loader(self):
        return self.normal_train_loader, self.normal_val_loader, self.normal_test_loader

    def decay_data_loader(self):
        return self.decay_train_loader, self.decay_val_loader, self.decay_test_loader


    def get_ml_dataset(self):
        x = self.x[self.index_]
        y = self.y[self.index_]


        train_instances = int(x.shape[0] * self.train_frac)
        dev_instances = int(self.x.shape[0] * self.dev_frac)
        test_instances = int(self.x.shape[0] * self.test_frac)
        # print(f'#### Test instances for data #{test_instances} with batch {self.batch_size}')

        train_x = x[:train_instances]
        dev_x = x[train_instances:train_instances + dev_instances]
        test_x = x[-test_instances:]

        train_ys = y[:train_instances]
        dev_ys = y[train_instances:train_instances + dev_instances]
        test_ys = y[-test_instances:]

        train_x = np.reshape( train_x, ( train_instances,-1 ))

        dev_x  = np.reshape( dev_x, ( dev_instances, -1))
        # train_x =  np.concatenate([train_x,dev_x], axis=0)
        # train_ys = np.concatenate([x for x in self.train_ys])
        train_ys =  train_ys
        dev_ys =  dev_ys
        # dev_ys = np.concatenate([x for x in self.dev_ys])
        # train_ys = np.concatenate([train_ys, dev_ys], axis=0)

        # test_x = np.concatenate([x for x in self.test_x])
        test_x = np.reshape( test_x, ( test_instances, -1))
        test_ys =  test_ys #np.concatenate([x for x in self.test_ys])

        return train_x,dev_x,test_x, train_ys,dev_ys, test_ys

    def get_ml_test(self):
        _, _, test_x, _, _, test_y = self.get_ml_dataset()
        return test_x,test_y


# if __name__ == '__main__':
#     # d = MIMICDataLoader(84, 24,
#     #                     'C:\\Users\\sirva\\PycharmProjects\\RIMs_for_clinical_ML\\data\\mimic_iii\\test_dump'
#     #                     '\\decay_data_20926.npz')
#
#     torch.manual_seed(1048)
#     torch.cuda.manual_seed(1048)
#     np.random.seed(1048)
#
#     data = np.arange(10)
#     for i in range(3):
#         np.random.shuffle(data)
#         print(data)
    # train_x,dev_x,test_x, train_ys,dev_ys, test_ys = d.get_ml_dataset()
#     print(train_x.shape)
#     # for x, static, y in tr:
#     #     print(f'x:{x.size()},s:{static.size()}, y:{y.size()}')
#     #     # print(f'x:{x.size()},s:{static.size()}, mean:{x_mean.size()}, y:{y.size()}')
#     #     break
#     # for i, x,s, m,y in tqdm(ran(t)):
#     #     print(f'x:{x.size()},s:{s.size()}, m:{m.size()}, y:{y.size()}')
