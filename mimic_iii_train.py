import torch
from tqdm import tqdm
import pickle
import numpy as np
from src.model.mimic_model import MIMICModel
from src.model.mimic_lstm_model import MIMICLSTMModel
from src.model.mimic_gru_model import MIMICGRUModel
from src.utils.mimic_iii_data import MIMICIIIData
from src.utils.data_prep import MortalityDataPrep
from src.model.mimic_decay_model import MIMICDecayModel
from src.utils.mimic_iii_decay_data import MIMICDecayData

'''
!!!! Important !!!!!!!
Input arguments like model_type and cell type are crucial.
'''
args = {
    'epochs':100,
    'batch_size': 128,
    'input_size': 1, #automatically picked from data
    'model_type': 'RIMDecay', # type of model  RIM, LSTM, GRU
    'hidden_size': 100,
    'num_rims': 4,
    'rnn_cell': 'LSTM-D', # type of cell LSTM, or GRU
    'input_key_size': 128,
    'input_value_size': 128,
    'input_query_size': 128,
    'num_input_heads': 2,
    'input_dropout': 0.05,
    'comm_key_size': 64,
    'comm_value_size': 100,
    'comm_query_size': 64,
    'num_comm_heads': 2,
    'comm_dropout': 0.05,
    'active_rims': 2, 
    'mask': False,
    'mask_size': 104,
    'delta_size': 104,
    'static_features':17, #automatically picked from data
    'need_data_preprocessing': False,
    'raw_data_file_path' :'data/mimic_iii/curated_30k/all_hourly_data_30000.pkl',
    'processed_data_path':'data/mimic_iii/test_dump',
    'input_file_path':'data/mimic_iii/test_dump/decay_data_20926.npz'
}

torch.manual_seed(10)
np.random.seed(10)
torch.cuda.manual_seed(10)


save_dir = 'mimic/models'
log_dir = 'mimic/logs'


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if args['model_type'] == 'RIMDecay':
    args['mask'] = True

# Data preprocessing
if(args['need_data_preprocessing']):
    prep_data = MortalityDataPrep(args['raw_data_file_path'])
    if args['mask']:
        args['input_file_path'] = prep_data.preprocess_decay(True, args['processed_data_path'])
    else:
        _, _, _, args['input_file_path'] = prep_data.preprocess(True, args['processed_data_path'])
        del _

# data loader
if args['model_type'] == 'RIMDecay':
    data = MIMICDecayData(args['batch_size'], 24, args['input_file_path'])
    args['input_size'] = data.input_size
    args['static_features'] = data.statics_size
    args['mask_size'] = data.input_size
    args['delta_size'] = data.input_size
    args['hidden_size'] = data.input_size
    args['comm_value_size'] = data.input_size
else:
    data = MIMICIIIData(args['batch_size'], 24, args['input_file_path'], args['mask'])
    args['input_size'] = data.input_size()
    args['static_features'] = data.static_features_size()
    args['mask_size'] = data.input_size()
    args['delta_size'] = data.input_size()



class RIMTrainer:
    def __init__(self, args, data_obj):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = MIMICModel(args).to(self.device)
        self.epochs = args['epochs']
        self.data_obj = data_obj

        print(f'Model: \n {self.model}')



    def eval(self):
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(self.data_obj.valid_len())):

                x, y, statics = self.data_obj.valid_get(i)
                x = self.model.to_device(x)
                statics = self.model.to_device(statics)
                y = self.model.to_device(y)

                predictions = self.model(x, statics)

                probs = torch.round(torch.sigmoid(predictions))
                correct = probs.view(-1) == y
                accuracy += correct.sum().item()
        
        accuracy /= self.data_obj.dev_instances
        return accuracy

    def test(self):
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(self.data_obj.test_len())):
                x, y, statics = self.data_obj.test_get(i)
                x = self.model.to_device(x)
                statics = self.model.to_device(statics)
                y = self.model.to_device(y)

                predictions = self.model(x, statics)
                
                probs = torch.round(torch.sigmoid(predictions))
                correct = probs.view(-1) == y
                accuracy += correct.sum().item()
        
        accuracy /= self.data_obj.test_instances
        return accuracy



    def train(self):
        acc = []
        train_acc = []
        test_acc = []
        loss_stats = []
        ctr = 0
        start_epochs = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)
        print(f"RIM Training, Validating, and Testing: {args['model_type']} model with {args['rnn_cell']} cell ")
        for epoch in range(start_epochs, self.epochs):
            print(f'EPOCH: {epoch +1}')
            epoch_loss = 0.0
            iter_ctr = 0.0
            t_accuracy = 0
            norm = 0

            self.model.train()
            for i in tqdm(range(self.data_obj.train_len())):
                iter_ctr += 1

                x, y, statics = self.data_obj.train_get(i)
                x = self.model.to_device(x)
                statics = self.model.to_device(statics)
                y = self.model.to_device(y)

                output, l = self.model(x, statics, y)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                norm += self.model.grad_norm()

                epoch_loss += l.item()
                predictions = torch.round(output)
                correct = predictions.view(-1) == y.long()
                t_accuracy += correct.sum().item()

                ctr += 1

            validation_accuracy = self.eval()
            test_accuracy = self.test()

            print(f'epoch loss: {epoch_loss}, taining accuracy: {t_accuracy/self.data_obj.train_instances}, validation accuracy: {validation_accuracy}, Test accuracy: {test_accuracy}')
            
            print("saving the models state...")
            model_state = {
                'net': self.model.state_dict(),
                'epochs': epoch,
                'ctr': ctr,
                'args':args
            }
            with open(f"{save_dir}/{args['model_type']}_{args['rnn_cell']}_model.pt", 'wb') as f:
                torch.save(model_state, f)

            

            loss_stats.append((ctr,epoch_loss/iter_ctr))
            acc.append((epoch,(validation_accuracy)))
            train_acc.append((epoch, (t_accuracy/self.data_obj.train_instances)))
            test_acc.append((epoch, (test_accuracy)))

            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_lossstats.pickle",'wb') as f:
                pickle.dump(loss_stats,f)
            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_accstats.pickle",'wb') as f:
                pickle.dump(acc,f)
            
            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_train_acc.pickle",'wb') as f:
                pickle.dump(train_acc,f)
            
            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_test_acc.pickle", 'wb') as f:
                pickle.dump(test_acc, f)


class RIMDEcayTrainer:
    def __init__(self, args, data_obj):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = MIMICDecayModel(args).to(self.device)
        self.epochs = args['epochs'] 
        self.data_obj = data_obj
        self.train_loader, self.val_loader, self.test_loader = data_obj.data_loader()
        print(f'Model: \n {self.model}')
        
    def eval(self):
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for x, static, x_mean, y in self.val_loader:
                # try: 
                #     x, static, x_mean, y = next(self.val_loader)
                # except StopIteration:
                #     self.val_loader = iter(self.val_loader)
                #     x, static, x_mean, y = next(self.val_loader)

                static = static.to(self.device)
                x_mask = x[:,1,:,:].to(self.device)
                delta = x[:,2,:,:].to(self.device)
                x_mean = x_mean.to(self.device)
                x_last_ob = x[:,3,:,:].to(self.device)
                x = x[:,0,:,:].to(self.device)

                y = y.to(self.device)

                predictions = self.model(x, static, x_mask, delta, x_last_ob, x_mean)
                probs = torch.round(torch.sigmoid(predictions))
                correct = probs.view(-1) == y
                accuracy += correct.sum().item()
        # todo compute accuracy
        accuracy /= self.data_obj.dev_instances
        return accuracy

    def test(self):
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for x, static, x_mean, y in self.test_loader:
                # try: 
                #     x, static, x_mean, y = next(self.test_loader)
                # except StopIteration:
                #     self.test_loader = iter(self.test_loader)
                #     x, static, x_mean, y = next(self.test_loader)

                static = static.to(self.device)
                x_mask = x[:,1,:,:].to(self.device)
                delta = x[:,2,:,:].to(self.device)
                x_mean = x_mean.to(self.device)
                x_last_ob = x[:,3,:,:].to(self.device)
                x = x[:,0,:,:].to(self.device)

                y = y.to(self.device)

                predictions = self.model(x, static, x_mask, delta, x_last_ob, x_mean)
                probs = torch.round(torch.sigmoid(predictions))
                correct = probs.view(-1) == y
                accuracy += correct.sum().item()
        # todo compute accuracy
        accuracy /= self.data_obj.test_instances
        return accuracy

    def train(self):
        acc = []
        train_acc = []
        test_acc = []
        loss_stats = []
        ctr = 0
        start_epochs = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.00001)
        print(f"RIM Decay Training, Validating, and Testing: {args['model_type']} model with {args['rnn_cell']} cell ")
        for epoch in range(start_epochs, self.epochs):
            print(f' EPOCH: {epoch +1}')
            epoch_loss = 0.0
            iter_ctr = 0.0
            t_accuracy = 0
            norm = 0

            self.model.train()
            for x, static, x_mean, y in self.train_loader:
                iter_ctr += 1
                # try: 
                #     x, static, x_mean, y = next(self.train_loader)
                # except StopIteration:
                #     self.train_loader = iter(self.train_loader)
                #     x, static, x_mean, y = next(self.train_loader)

                static = static.to(self.device)
                x_mask = x[:,1,:,:].to(self.device)
                delta = x[:,2,:,:].to(self.device)
                x_mean = x_mean.to(self.device)
                x_last_ob = x[:,3,:,:].to(self.device)
                x = x[:,0,:,:].to(self.device)

                y = y.to(self.device)

                output, l = self.model(x, static, x_mask, delta, x_last_ob, x_mean, y)
            
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                norm += self.model.grad_norm()

                epoch_loss += l.item()
                predictions = torch.round(output)
                correct = predictions.view(-1) == y.long()
                t_accuracy += correct.sum().item()

                ctr += 1
        
            validation_accuracy = self.eval()
            test_accuracy = self.test()

            print(f'epoch loss: {epoch_loss}, taining accuracy: {t_accuracy/self.data_obj.train_instances}, validation accuracy: {validation_accuracy}, Test accuracy: {test_accuracy}')
            
            print("saving the models state...")
            model_state = {
                'net': self.model.state_dict(),
                'epochs': epoch,
                'ctr': ctr,
                'args': args
            }
            with open(f"{save_dir}/{args['model_type']}_{args['rnn_cell']}_model.pt", 'wb') as f:
                torch.save(model_state, f)

            

            loss_stats.append((ctr,epoch_loss/iter_ctr))
            acc.append((epoch,(validation_accuracy)))
            train_acc.append((epoch, (t_accuracy/data.train_instances)))
            test_acc.append((epoch, (test_accuracy)))

            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_lossstats.pickle",'wb') as f:
                pickle.dump(loss_stats,f)
            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_accstats.pickle",'wb') as f:
                pickle.dump(acc,f)
            
            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_train_acc.pickle",'wb') as f:
                pickle.dump(train_acc,f)
            
            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_test_acc.pickle", 'wb') as f:
                pickle.dump(test_acc, f)         

        

if args['model_type'] == 'LSTM':
    model = MIMICLSTMModel(args)
elif args['model_type'] == 'GRU':
    model = MIMICGRUModel(args)
elif args['model_type'] == 'RIMDecay':
    model = RIMDEcayTrainer(args, data)
    model.train()
else:
    model = RIMTrainer(args, data)
    model.train()

# train_model(model, args['epochs'], data)

