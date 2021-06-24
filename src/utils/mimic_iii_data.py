import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class MIMICIIIData:
    def __init__(self, batch_size, window_size, file_path, mask=False, train_frac=0.7, dev_frac= 0.15, test_frac=0.15):
        self.batch_size = batch_size
        self.window_size = window_size
        self.apply_mask = mask

        all_data = np.load(file_path,allow_pickle=True)
        self.x = all_data['x']
        self.y = all_data['y']
        self.statics = all_data['statics']

        del all_data

        self.statics_size = self.statics.shape[-1]
        self.input_size  = self.x.shape[-1]

        self.x = np.reshape(self.x, (self.y.shape[0], self.window_size, -1))
        self.y = np.squeeze(self.y, axis=1)

        self.train_instances = int(self.x.shape[0] *  train_frac)
        self.dev_instances = int(self.x.shape[0] * dev_frac)
        self.test_instances = int(self.x.shape[0] * test_frac)

        index_ = np.arange(self.x.shape[0], dtype=int)
        np.random.seed(1024)
        np.random.shuffle(index_)

        self.x = self.x[index_]
        self.y = self.y[index_]
        self.statics = self.statics[index_]


        self.train_x = self.x[:self.train_instances]
        self.dev_x = self.x[self.train_instances:self.train_instances + self.dev_instances]
        self.test_x = self.x[-self.test_instances:]
        
        self.train_ys = self.y[:self.train_instances]
        self.dev_ys = self.y[self.train_instances:self.train_instances + self.dev_instances]
        self.test_ys = self.y[-self.test_instances:]

        self.train_statics = self.statics[:self.train_instances]
        self.dev_statics = self.statics[self.train_instances:self.train_instances + self.dev_instances]
        self.test_statics = self.statics[-self.test_instances:]

    def get_test_data(self):
        test_data, test_label =torch.from_numpy(self.test_x), torch.from_numpy(self.test_ys)
        test_static = torch.from_numpy(self.test_statics)
        return (test_data, test_static, test_label)

    def data_loader(self):
        train_data, train_label = torch.from_numpy(self.train_x), torch.from_numpy(self.train_ys)
        val_data, val_label = torch.from_numpy(self.dev_x), torch.from_numpy(self.dev_ys)
        test_data, test_label = torch.from_numpy(self.test_x), torch.from_numpy(self.test_ys)

        train_static, val_static= torch.from_numpy(self.train_statics), torch.from_numpy(self.dev_statics)
        test_static = torch.from_numpy(self.test_statics)

        train_dataset = TensorDataset(train_data, train_static, train_label)
        val_dataset = TensorDataset(val_data, val_static, val_label)
        test_dataset = TensorDataset(test_data, test_static, test_label)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    def train_len(self):
        return len(self.train_ys)
    
    def valid_len(self):
        return len(self.dev_ys)
    
    def test_len(self):
        return len(self.test_ys)

    def static_features_size(self):
        return self.statics.shape[1]

    def train_get(self, i):
        if self.apply_mask:
            return self.train_x[i], self.train_ys[i], self.train_statics[i], self.train_mask[i], self.train_delta[i], self.train_x_mean[i], self.train_last_observed[i]
        return self.train_x[i], self.train_ys[i], self.train_statics[i]

    def valid_get(self, i):
        if self.apply_mask:
            return self.dev_x[i], self.dev_ys[i], self.dev_statics[i], self.dev_mask[i], self.dev_delta[i], self.dev_x_mean[i], self.dev_last_observed[i]
        return self.dev_x[i], self.dev_ys[i], self.dev_statics[i]
    
    def test_get(self, i):
        if self.apply_mask:
            return self.test_x[i], self.test_ys[i], self.test_statics[i], self.test_mask[i], self.test_delta[i], self.test_x_mean[i], self.test_last_observed[i]
        return self.test_x[i], self.test_ys[i], self.test_statics[i]

    def total_instances(self):
        return self.train_instances, self.dev_instances, self.test_instances
    
    def input_size(self):
        return self.x.shape[2]
    def statics_size(self):
        return self.statics.shape[2]

    """
    SK-Learn dataset prepartion 
    concatenated the datasets and returned two train and test set for both input and target features
    """
    def get_ml_dataset(self):

        train_x = np.reshape(self.train_x, (self.train_instances,-1 ))

        dev_x  = np.reshape(self.dev_x, (self.dev_instances, -1))
        # train_x =  np.concatenate([train_x,dev_x], axis=0)
        # train_ys = np.concatenate([x for x in self.train_ys])
        train_ys = self.train_ys
        dev_ys = self.dev_ys
        # dev_ys = np.concatenate([x for x in self.dev_ys])
        # train_ys = np.concatenate([train_ys, dev_ys], axis=0)

        # test_x = np.concatenate([x for x in self.test_x])
        test_x = np.reshape(self.test_x, (self.test_instances, -1))
        test_ys = self.test_ys #np.concatenate([x for x in self.test_ys])

        return train_x,dev_x,test_x, train_ys,dev_ys, test_ys

    def get_ml_test(self):
        _, _, test_x, _, _, test_y = self.get_ml_dataset()
        return test_x,test_y

    def variable_feature_size(self):
        return self.train_x[0][0].shape[0],self.train_x[0][0].shape[1]

# if __name__ == '__main__':
#     print(os.curdir)
#
#     file_path =  '../../data/x_y_statics_20926.npz'
#     data = MIMICIIIData(10, 24, file_path, mask=False)
#     x, y, s = data.train_get(1)
#     print(data.static_features_size())
#     print(data.train_len())
#     x_train, x_test, y_train, y_test = data.get_sk_dataset()
#
#     # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     # x = torch.from_numpy(x).to(device)


# if __name__ =='__main__':
#     d = MIMICIIIData(64, 24 , './data/mimic_iii/test_dump/decay_data_20926.npz')
#     t_l,vl,tl = d.data_loader()
#     # print(x)
#     for x, s,y in t_l:
#         print(f'x:{x.size()},s:{s.size()}, y:{y.size()}')