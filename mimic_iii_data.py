import numpy as np

class MIMICIIIData:
    def __init__(self, batch_size, window_size, file_path, train_frac=0.7, dev_frac= 0.15, test_frac=0.15):
        self.batch_size = batch_size
        self.window_size = window_size

        all_data = np.load(file_path)
        self.x = all_data['x']
        self.y = all_data['y']
        self.statics = all_data['statics']
        del all_data

        self.x = np.reshape(self.x, (self.y.shape[0], self.window_size, -1))
        self.y = np.squeeze(self.y, axis=1)

        self.train_instances = int(self.x.shape[0] *  train_frac)
        self.dev_instances = int(self.x.shape[0] * dev_frac)
        self.test_instances = int(self.x.shape[0] * test_frac)

        self.train_x = self.x[:self.train_instances, :, :]
        self.dev_x = self.x[self.train_instances:self.train_instances + self.dev_instances, : , :]
        self.test_x = self.x[-self.test_instances: , :, : ] 
        
        self.train_ys = self.y[:self.train_instances]
        self.dev_ys = self.y[self.train_instances:self.train_instances + self.dev_instances]
        self.test_ys = self.y[-self.test_instances:]

        self.train_statics = self.statics[:self.train_instances, :]
        self.dev_statics = self.statics[self.train_instances:self.train_instances + self.dev_instances, :]
        self.test_statics = self.statics[-self.test_instances:, :]

        # batchify 
        self.train_x = [self.train_x[i:i + self.batch_size] for i in range(0, self.train_x.shape[0], self.batch_size)]
        self.train_ys = [self.train_ys[i:i + self.batch_size] for i in range(0, self.train_ys.shape[0], self.batch_size)]
        self.train_statics = [self.train_statics[i:i+self.batch_size] for i in range(0, self.train_statics.shape[0], self.batch_size)]

        self.dev_x = [self.dev_x[i:i+32] for i in range(0, self.dev_x.shape[0], 32)]
        self.dev_ys = [self.dev_ys[i:i+32] for i in range(0, self.dev_ys.shape[0], 32)]
        self.dev_statics = [self.dev_statics[i:i+32] for i in range(0, self.dev_statics.shape[0], 32)]

        self.test_x = [self.test_x[i:i+32] for i in range(0, self.test_x.shape[0], 32)]
        self.test_ys = [self.test_ys[i:i+32] for i in range(0, self.test_ys.shape[0], 32)]
        self.test_statics = [self.test_statics[i:i+32] for i in range(0, self.test_statics.shape[0], 32)]



    def train_len(self):
        return len(self.train_ys)
    
    def valid_len(self):
        return len(self.dev_ys)
    
    def test_len(self):
        return len(self.test_ys)

    def static_features_size(self):
        return self.statics.shape[1]

    def train_get(self, i):
        return self.train_x[i], self.train_ys[i], self.train_statics[i]
    
    def valid_get(self, i):
        return self.dev_x[i], self.dev_ys[i], self.dev_statics[i]
    
    def test_get(self, i):
        return self.test_x[i], self.test_ys[i], self.test_statics[i]

    def total_instances(self):
        return self.train_instances, self.dev_instances, self.test_instances
    
    def input_size(self):
        return self.x.shape[2]

        
# if __name__ == '__main__':
#     data = MIMICIIIData(25, 24)
#     x, y, s = data.train_get(1)
#     print(data.static_features_size())
#     print(data.train_len())
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     x = torch.from_numpy(x).to(device)
