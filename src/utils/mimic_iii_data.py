import os.path

import numpy as np
from numpy import random

class MIMICIIIData:
    def __init__(self, batch_size, window_size, file_path, mask=True, train_frac=0.7, dev_frac= 0.15, test_frac=0.15):
        self.batch_size = batch_size
        self.window_size = window_size
        self.apply_mask = mask

        all_data = np.load(file_path)
        self.x = all_data['x']
        self.y = all_data['y']
        self.statics = all_data['statics']

        if self.apply_mask:
            self.mask = all_data['mask']
            self.delta = all_data['delta']
            self.x_mean = all_data['xmean']
            self.last_observed = all_data['lastob']

        del all_data

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


        self.train_x = self.x[:self.train_instances, :, :]
        self.dev_x = self.x[self.train_instances:self.train_instances + self.dev_instances, : , :]
        self.test_x = self.x[-self.test_instances: , :, : ] 
        
        self.train_ys = self.y[:self.train_instances]
        self.dev_ys = self.y[self.train_instances:self.train_instances + self.dev_instances]
        self.test_ys = self.y[-self.test_instances:]

        self.train_statics = self.statics[:self.train_instances, :]
        self.dev_statics = self.statics[self.train_instances:self.train_instances + self.dev_instances, :]
        self.test_statics = self.statics[-self.test_instances:, :]

        if self.apply_mask:
            self.mask = np.reshape(self.mask, (self.y.shape[0], self.window_size, -1))
            self.delta = np.reshape(self.delta, (self.y.shape[0], self.window_size, -1))
            self.last_observed = np.reshape(self.last_observed, (self.y.shape[0], self.window_size, -1))

            self.mask = self.mask[index_]
            self.delta = self.delta[index_]
            self.last_observed = self.last_observed[index_]
            self.x_mean = self.x_mean[index_]

            self.train_mask = self.mask[:self.train_instances, :, :]
            self.dev_mask = self.mask[self.train_instances:self.train_instances + self.dev_instances, : , :]
            self.test_mask = self.mask[-self.test_instances: , :, : ]

            self.train_delta = self.delta[:self.train_instances, :, :]
            self.dev_delta = self.delta[self.train_instances:self.train_instances + self.dev_instances, : , :]
            self.test_delta = self.delta[-self.test_instances: , :, : ]

            self.train_last_observed = self.last_observed[:self.train_instances, :, :]
            self.dev_last_observed = self.last_observed[self.train_instances:self.train_instances + self.dev_instances, : , :]
            self.test_last_observed = self.last_observed[-self.test_instances: , :, : ]

            self.train_x_mean = self.x_mean[:self.train_instances, :, :]
            self.dev_x_mean = self.x_mean[self.train_instances:self.train_instances + self.dev_instances, : , :]
            self.test_x_mean = self.x_mean[-self.test_instances: , :, : ]

            #batchify 
            self.train_mask = [self.train_mask[i:i + self.batch_size] for i in range(0, self.train_mask.shape[0], self.batch_size)]
            self.train_delta = [self.train_delta[i:i+self.batch_size] for i in range(0, self.train_delta.shape[0], self.batch_size)]
            self.train_x_mean = [self.train_x_mean[i:i+self.batch_size] for i in range(0, self.train_x_mean.shape[0], self.batch_size)]
            self.train_last_observed = [self.train_last_observed[i:i+self.batch_size] for i in range(0, self.train_last_observed.shape[0], self.batch_size)]

            self.dev_mask = [self.dev_mask[i:i+self.batch_size] for i in range(0, self.dev_mask.shape[0], self.batch_size)]
            self.dev_delta = [self.dev_delta[i:i+self.batch_size] for i in range(0, self.dev_delta.shape[0], self.batch_size)]
            self.dev_x_mean = [self.dev_x_mean[i:i+self.batch_size] for i in range(0, self.dev_x_mean.shape[0], self.batch_size)]
            self.last_observed = [self.dev_last_observed[i:i+self.batch_size] for i in range(0, self.dev_last_observed.shape[0], self.batch_size)]

            self.test_mask = [self.test_mask[i:i+self.batch_size] for i in range(0, self.test_mask.shape[0], self.batch_size)]
            self.test_delta = [self.test_delta[i:i+self.batch_size] for i in range(0, self.test_delta.shape[0], self.batch_size)]
            self.test_x_mean = [self.test_x_mean[i:i+self.batch_size] for i in range(0, self.test_x_mean.shape[0], self.batch_size)]
            self.test_last_observed = [self.test_last_observed[i:i+self.batch_size] for i in range(0, self.test_last_observed.shape[0], self.batch_size)]

        # batchify

        self.train_x = [self.train_x[i:i + self.batch_size] for i in range(0, self.train_x.shape[0], self.batch_size)]
        self.train_ys = [self.train_ys[i:i + self.batch_size] for i in range(0, self.train_ys.shape[0], self.batch_size)]
        self.train_statics = [self.train_statics[i:i+self.batch_size] for i in range(0, self.train_statics.shape[0], self.batch_size)]

        self.dev_x = [self.dev_x[i:i+self.batch_size] for i in range(0, self.dev_x.shape[0], self.batch_size)]
        self.dev_ys = [self.dev_ys[i:i+self.batch_size] for i in range(0, self.dev_ys.shape[0], self.batch_size)]
        self.dev_statics = [self.dev_statics[i:i+self.batch_size] for i in range(0, self.dev_statics.shape[0], self.batch_size)]

        self.test_x = [self.test_x[i:i+self.batch_size] for i in range(0, self.test_x.shape[0], self.batch_size)]
        self.test_ys = [self.test_ys[i:i+self.batch_size] for i in range(0, self.test_ys.shape[0], self.batch_size)]
        self.test_statics = [self.test_statics[i:i+self.batch_size] for i in range(0, self.test_statics.shape[0], self.batch_size)]



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
    """
    SK-Learn dataset prepartion 
    concatenated the datasets and returned two train and test set for both input and target features
    """
    def get_sk_dataset(self):
        train_x =  np.concatenate([x for x in self.train_x])
        train_x = np.reshape(train_x, (self.train_instances,-1 ))
        dev_x =  np.concatenate([x for x in self.dev_x])
        dev_x  = np.reshape(dev_x, (self.dev_instances, -1))
        # train_x =  np.concatenate([train_x,dev_x], axis=0)


        train_ys = np.concatenate([x for x in self.train_ys])
        dev_ys = np.concatenate([x for x in self.dev_ys])
        # train_ys = np.concatenate([train_ys, dev_ys], axis=0)

        test_x = np.concatenate([x for x in self.test_x])
        test_x = np.reshape(test_x, (self.test_instances, -1))

        test_ys = np.concatenate([x for x in self.test_ys])

        return train_x,dev_x,test_x, train_ys,dev_ys, test_ys


    def variable_feature_size(self):
        return self.train_x[0][0].shape[0],self.train_x[0][0].shape[1]
if __name__ == '__main__':
    print(os.curdir)

    file_path =  '../../data/x_y_statics_20926.npz'
    data = MIMICIIIData(10, 24, file_path, mask=False)
    x, y, s = data.train_get(1)
    print(data.static_features_size())
    print(data.train_len())
    x_train, x_test, y_train, y_test = data.get_sk_dataset()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # x = torch.from_numpy(x).to(device)
