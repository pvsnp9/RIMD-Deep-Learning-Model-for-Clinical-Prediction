import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader


class Mimic_Kfold_Data():

    def __init__(self, window_size, file_path,k_fold =5, train_frac=0.7, dev_frac=0.15, test_frac=0.15,
                 balance=False):
        self.window_size = window_size
        self.train_frac = train_frac
        self.dev_frac = dev_frac
        self.test_frac = test_frac
        self.k_fold = k_fold
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
        # np.random.shuffle(self.index_)
        skf = StratifiedKFold(n_splits=k_fold, random_state=1024, shuffle=True)

        folds = []
        for train_index, dev_test_index in skf.split(self.x, self.y):
            print("TRAIN:", len(train_index), "TEST:", len(dev_test_index))
            sp1 = len(dev_test_index) // 2
            dev_test_x = self.x[dev_test_index]
            dev_test_y = self.y[dev_test_index]

            skf2 = StratifiedKFold(n_splits=2, random_state=48, shuffle=True)
            dev_index, test_index = skf2.split(dev_test_x, dev_test_y)

            dev_x = dev_test_x[dev_index[0]]
            dev_y = dev_test_y[dev_index[0]]
            test_x = dev_test_x[test_index[0]]
            test_y = dev_test_y[test_index[0]]
            (_, dev_count) = np.unique(dev_y, return_counts=True)
            (_, test_count) = np.unique(test_y, return_counts=True)
            print(f'dev set:{dev_count}, test set: {test_count}')
            dev_x_index = dev_test_index[dev_index[0]]
            test_x_index = dev_test_index[dev_index[1]]
            folds.append([train_index, dev_x_index, test_x_index])
        self.folds = folds

    def get_folds(self):
        return self.folds

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
        train_indicies, val_indicies, test_indicies = self.folds[self.fold]

        train_x = self.x[train_indicies]
        dev_x = self.x[val_indicies]
        test_x = self.x[test_indicies]

        train_ys = self.y[train_indicies]
        dev_ys = self.y[val_indicies]
        test_ys = self.y[test_indicies]

        train_statics = self.statics[train_indicies]
        dev_statics = self.statics[val_indicies]
        test_statics = self.statics[test_indicies]

        # counting nums
        (_, counts_all) = np.unique(self.y, return_counts=True)
        (_, counts_train) = np.unique(train_ys, return_counts=True)
        (_, counts_val) = np.unique(dev_ys, return_counts=True)
        (_, counts_test) = np.unique(test_ys, return_counts=True)

        print(
            f"***************** COUNTS: [0,1]s ==> total samples:{counts_all}, train_samples:{counts_train}, val_samples:{counts_val} test_sample:{counts_test} ******************************")

        train_data, train_label = torch.from_numpy(train_x), torch.from_numpy(train_ys)
        val_data, val_label = torch.from_numpy(dev_x), torch.from_numpy(dev_ys)
        test_data, test_label = torch.from_numpy(test_x), torch.from_numpy(test_ys)

        train_static, val_static = torch.from_numpy(train_statics), torch.from_numpy(dev_statics)
        test_static = torch.from_numpy(test_statics)

        train_dataset = TensorDataset(train_data, train_static, train_label)
        val_dataset = TensorDataset(val_data, val_static, val_label)
        test_dataset = TensorDataset(test_data, test_static, test_label)

        self.normal_train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.normal_val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        self.normal_test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

    def prepare_decay_loader(self):

        train_indicies, val_indicies, test_indicies = self.folds[self.fold]
        # expanding dims to concatenate
        x = np.expand_dims(self.x, axis=1)
        mask = np.expand_dims(self.mask, axis=1)
        delta = np.expand_dims(self.delta, axis=1)
        last_observed = np.expand_dims(self.last_observed, axis=1)

        data_agg = np.concatenate((x, mask, delta, last_observed), axis=1)

        train_data, train_label = data_agg[train_indicies], self.y[train_indicies]
        valid_data, valid_label = data_agg[val_indicies], self.y[val_indicies]
        test_data, test_label = data_agg[test_indicies], self.y[test_indicies]

        # counting nums
        (_, counts_all) = np.unique(self.y, return_counts=True)
        (_, counts_train) = np.unique(train_label, return_counts=True)
        (_, counts_val) = np.unique(valid_label, return_counts=True)
        (_, counts_test) = np.unique(test_label, return_counts=True)

        print(
            f"***************** COUNTS: [0,1]s ==> total samples:{counts_all}, train_samples:{counts_train}, val_samples:{counts_val} test_sample:{counts_test} ***************************")
        print(
            f"***************** COUNTS: [0,1]s ==> total samples:{counts_all}, train_samples:{counts_train[0] / sum(counts_train)},{counts_train[1] / sum(counts_train)}, val_samples:{counts_val[0] / sum(counts_val)},{counts_val[1] / sum(counts_val)}test_sample:{counts_test[0] / sum(counts_test)},{counts_test[1] / sum(counts_test)} ***************************")

        train_static = self.statics[train_indicies]
        valid_static = self.statics[val_indicies]
        test_static = self.statics[test_indicies]

        train_x_mean = self.x_mean[train_indicies]
        valid_x_mean = self.x_mean[val_indicies]
        test_x_mean = self.x_mean[test_indicies]

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

        train_indicies, val_indicies, test_indicies = self.folds[self.fold]

        train_x = self.x[train_indicies]
        val_x = self.x[val_indicies]
        test_x = self.x[test_indicies]

        train_ys = self.y[train_indicies]
        val_ys = self.y[val_indicies]
        test_ys = self.y[test_indicies]

        train_x = np.reshape(train_x, (len(train_indicies), -1))

        val_x = np.reshape(val_x, (len(val_indicies), -1))
        # train_x =  np.concatenate([train_x,dev_x], axis=0)
        # train_ys = np.concatenate([x for x in self.train_ys])
        train_ys = train_ys
        val_ys = val_ys
        # dev_ys = np.concatenate([x for x in self.dev_ys])
        # train_ys = np.concatenate([train_ys, dev_ys], axis=0)

        # test_x = np.concatenate([x for x in self.test_x])
        test_x = np.reshape(test_x, (len(test_indicies), -1))
        test_ys = test_ys  # np.concatenate([x for x in self.test_ys])

        return train_x, val_x, test_x, train_ys, val_ys, test_ys

    def get_ml_test(self):
        _, _, test_x, _, _, test_y = self.get_ml_dataset()
        return test_x, test_y

    def set_fold(self, fold):
        self.fold = fold


if __name__ == '__main__':

    d = Mimic_Kfold_Data(24,'../../data/mimic_iii/test_dump/decay_data_20926.npz')
    for fold_index in range(len(d.get_folds())):
        d.prepare_data_loader(fold_index, 'lalaal', 128)
        _, _, _ = d.normal_data_loader()
