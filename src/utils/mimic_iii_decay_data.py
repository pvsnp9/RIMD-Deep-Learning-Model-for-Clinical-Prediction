import math
import numpy as np
from numpy.core.defchararray import index
from numpy.core.fromnumeric import shape
import torch
from torch.utils.data import DataLoader, TensorDataset

class MIMICDecayData:
    def __init__(self, batch_size, window_size, file_path, train_frac=0.7, dev_frac= 0.15, test_frac=0.15 , balance=False):
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

        self.x = np.reshape(self.x, (self.y.shape[0], self.window_size, -1))
        self.y = np.squeeze(self.y, axis=1)

        self.mask = np.reshape(self.mask, (self.y.shape[0], self.window_size, -1))
        self.delta = np.reshape(self.delta, (self.y.shape[0], self.window_size, -1))
        self.last_observed = np.reshape(self.last_observed, (self.y.shape[0], self.window_size, -1))

        # delta normalization for each patients
        delta_mean = np.amax(self.delta, axis=1)
        for i in range(self.delta.shape[0]):
            self.delta[i] = np.divide(self.delta[i], delta_mean[i], out=np.zeros_like(self.delta[i]), where=delta_mean[i]!=0)


        del all_data, delta_mean

        '''
        Undersampling
        '''
        # np.random.seed(1024)

        zero_index = np.squeeze(np.where(self.y == 0) )
        #ToDO we may need to set Seed !
        if balance:
            np.random.shuffle(zero_index)
            ones_index = np.squeeze(np.where(self.y == 1) )
            zero_index = zero_index[:math.ceil(0.5 * len(zero_index))]
            new_index = np.concatenate((ones_index,zero_index))

            self.x = self.x[new_index]
            self.y = self.y[new_index]
            self.statics = self.statics[new_index]
            self.mask = self.mask[new_index]
            self.delta = self.delta[new_index]
            self.last_observed = self.last_observed[new_index]
            self.x_mean = self.x_mean[new_index]


        index_ = np.arange(self.x.shape[0], dtype = int)
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

        self.prepare_dataloader()


    def get_test_data(self):
        test_data, test_label = torch.from_numpy(self.test_data), torch.from_numpy(self.test_label)
        test_static = torch.from_numpy(self.test_static)
        test_x_mean = torch.from_numpy(self.test_x_mean)

        return (test_data, test_static, test_x_mean, test_label)

    def prepare_dataloader(self):
        train_data, train_label = torch.from_numpy(self.train_data), torch.from_numpy(self.train_label)

        valid_data, valid_label = torch.from_numpy(self.valid_data), torch.from_numpy(self.valid_label)
        test_data, test_label = torch.from_numpy(self.test_data), torch.from_numpy(self.test_label)

        train_static = torch.from_numpy(self.train_static)
        valid_static = torch.from_numpy(self.valid_static)
        test_static = torch.from_numpy(self.test_static)

        train_x_mean = torch.from_numpy(self.train_x_mean)
        test_x_mean = torch.from_numpy(self.test_x_mean)
        valid_x_mean = torch.from_numpy(self.valid_x_mean)

        train_dataset = TensorDataset(train_data, train_static, train_x_mean, train_label)
        val_dataset = TensorDataset(valid_data, valid_static, valid_x_mean, valid_label)
        test_dataset = TensorDataset(test_data, test_static, test_x_mean, test_label)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.val_loader   = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_loader  = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def data_loader(self):
        return self.train_loader, self.val_loader, self.test_loader

    def set_batch_size(self,batch_size):
        self.batch_size = batch_size

'''
non-iid data loader class
'''
class MIMICNonIidData:
    def __init__(self, file_path, number_of_instances=1000 ):
        self.file_path = file_path
        self.number_of_instances = number_of_instances
        self.data = None

    def __read_data(self):
        self.data = np.load(self.file_path, allow_pickle=True)
        # self.h18, self.h30, self.h36, self.h42, self.h48 = data['h_18'], data['h_30'], data['h_36'], data['h_42'], data['h_48']
    
    def get_data(self, hour=18):
        self.__read_data()
        if hour ==18:
            return self.__prepare_data(self.data['h_18'], 18)
        elif hour == 30:
            return self.__prepare_data(self.data['h_30'], 30)
        elif hour == 36:
            return self.__prepare_data(self.data['h_36'], 36)
        elif hour == 42:
            return self.__prepare_data(self.data['h_42'], 42)
        elif hour == 48:
            return self.__prepare_data(self.data['h_48'], 48)
        else:
            raise Exception(f'Data could not be found on {hour}-hr')
    
    def __prepare_data(self, hour_data, hour):
        x,y,statics, x_mean, x_mask, delta, last_ob = hour_data 
        del hour_data
        x= np.reshape(x, (y.shape[0], hour, -1))
        y = np.squeeze(y, axis=1)
        # class balanced data

        zero_index = np.squeeze(np.where(y == 0) )
        np.random.shuffle(zero_index)
        ones_index = np.squeeze(np.where(y == 1) )
        zero_index = zero_index[:math.floor(0.6 * self.number_of_instances)]
        ones_index = ones_index[-math.ceil(0.4 * self.number_of_instances):]
        new_index = np.concatenate((ones_index,zero_index))
        

        mask = np.reshape(x_mask, (y.shape[0], hour, -1))
        delta = np.reshape(delta, (y.shape[0], hour, -1))
        last_observed= np.reshape(last_ob, (y.shape[0], hour, -1))

        x = x[new_index]
        y = y[new_index]
        statics = statics[new_index]
        mask = mask[new_index]
        delta = delta[new_index]
        last_observed = last_observed[new_index]

        # delta normalization for each patients
        delta_mean = np.amax(delta, axis=1)
        for i in range(delta.shape[0]):
            delta[i] = np.divide(delta[i], delta_mean[i], out=np.zeros_like(delta[i]), where=delta_mean[i]!=0)

        '''Data shuffling'''        
        index_ = np.arange(self.number_of_instances, dtype=int)
        np.random.shuffle(index_)

        x = x[index_]
        y = y[index_]
        statics = statics[index_]
        mask = mask[index_]
        delta = delta[index_]
        last_observed = last_observed[index_]
        x_mean = x_mean[index_]

        # expanding dims to concatenate
        x = np.expand_dims(x, axis=1)
        mask = np.expand_dims(mask, axis=1)
        delta = np.expand_dims(delta, axis=1)
        last_observed = np.expand_dims(last_observed, axis=1)

        data_agg = np.concatenate((x, mask, delta, last_observed), axis=1)

        return (
            torch.from_numpy(data_agg),
            torch.from_numpy(statics),
            torch.from_numpy(x_mean),
            torch.from_numpy(y)
        )


# if __name__ =='__main__':
#     d = MIMICNonIidData('./data/mimic_iii/test_dump/non_iid_los_icu.npz')
#     x = d.get_data(48)
#     #x,y,z,t = d.get_test_data()
#     for num in range(len(x)):
#         print(torch.isnan(x[num]).any())
    # print(len(x))
    # for i, x,s, m,y in tqdm(ran(t)):
    #     print(f'x:{x.size()},s:{s.size()}, m:{m.size()}, y:{y.size()}')