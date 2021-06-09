import torch
from tqdm import tqdm
import pickle
import numpy as np
from src.model.mimic_model import MIMICModel
from src.model.mimic_lstm_model import MIMICLSTMModel
from src.model.mimic_gru_model import MIMICGRUModel
from src.utils.mimic_iii_data import MIMICIIIData
from src.utils.data_prep import MortalityDataPrep

'''
!!!! Important !!!!!!!
Input arguments like model_type and cell type are crucial.
'''
args = {
    'epochs':100,
    'batch_size': 64,
    'input_size': 1, #automatically picked from data
    'model_type': 'RIM', # type of model  RIM, LSTM, GRU
    'hidden_size': 100,
    'num_rims': 6,
    'rnn_cell': 'LSTM', # type of cell LSTM, or GRU
    'input_key_size': 64,
    'input_value_size': 400,
    'input_query_size': 64,
    'num_input_heads': 1,
    'input_dropout': 0.1,
    'comm_key_size': 32,
    'comm_value_size': 100,
    'comm_query_size': 32,
    'num_comm_heads': 2,
    'comm_dropout': 0.1,
    'active_rims': 4, 
    'static_features':17, #automatically picked from data
    'need_data_preprocessing': False,
    'raw_data_file_path' :'data/mimic_iii/test_dump/all_hourly_data.pkl',
    'processed_data_path':'data/mimic_iii/preprocessed/mortality_and_los/test',
    'input_file_path':'data/mimic_iii/preprocessed/mortality_and_los/decay_data_20926.npz'
}

torch.manual_seed(10)
np.random.seed(10)
torch.cuda.manual_seed(10)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data preprocessing
if(args['need_data_preprocessing']):
    prep_data = MortalityDataPrep(args['raw_data_file_path'])
    _, _, _, args['input_file_path'] = prep_data.preprocess(True, args['processed_data_path'])
    del _

# data loader
data = MIMICIIIData(args['batch_size'], 24, args['input_file_path'], True)
args['input_size'] = data.input_size()
args['static_features'] = data.static_features_size()

if args['model_type'] == 'LSTM':
    model = MIMICLSTMModel(args)
elif args['model_type'] == 'GRU':
    model = MIMICGRUModel(args)
else:
    model = MIMICModel(args)


model.to(device)
print(f'Model: \n {model}')
save_dir = 'mimic/models'
log_dir = 'mimic/logs'

def eval_model(model, data, data_getter_func):
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(data.valid_len())):
            x, y, statics = data_getter_func(i)
            x = model.to_device(x)
            y = model.to_device(y).float()
            statics = model.to_device(statics)

            predictions = model(x, statics)
            probs = torch.round(torch.sigmoid(predictions))
            correct = probs.view(-1) == y
            accuracy += correct.sum().item()
    
    accuracy /= data.dev_instances
    return accuracy

def test_model(model, data, data_getter_func):
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(data.test_len())):
            x, y, statics = data_getter_func(i)
            x = model.to_device(x)
            y = model.to_device(y).float()
            statics = model.to_device(statics)

            predictions = model(x, statics)
            probs = torch.round(torch.sigmoid(predictions))
            correct = probs.view(-1) == y
            accuracy += correct.sum().item()
    
    accuracy /= data.test_instances
    return accuracy



def train_model(model, epochs, data):
    acc = []
    train_acc = []
    test_acc = []
    loss_stats = []
    ctr = 0
    start_epochs = 0

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    print(f"Training, Validating, and Testing: {args['model_type']} model with {args['rnn_cell']} cell ")
    for epoch in range(start_epochs, epochs):
        print(f'EPOCH: {epoch +1}')
        epoch_loss = 0.0
        iter_ctr = 0.0
        t_accuracy = 0
        norm = 0

        model.train()
        for i in tqdm(range(data.train_len())):
            iter_ctr += 1
            x, y, static = data.train_get(i)
            x = model.to_device(x)
            static = model.to_device(static)
            y = model.to_device(y)

            output, l = model(x, static, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            norm += model.grad_norm()

            epoch_loss += l.item()
            predictions = torch.round(output)
            correct = predictions.view(-1) == y.long()
            t_accuracy += correct.sum().item()

            ctr += 1

        validation_accuracy = eval_model(model, data, data.valid_get)
        test_accuracy = test_model(model, data, data.test_get)

        print(f'epoch loss: {epoch_loss}, taining accuracy: {t_accuracy/data.train_instances}, validation accuracy: {validation_accuracy}, Test accuracy: {test_accuracy}, norm: {norm / iter_ctr}')
        
        print("saving the models state...")
        model_state = {
            'net': model.state_dict(),
            'epochs': epoch,
            'ctr': ctr
        }
        with open(f"{save_dir}/{args['model_type']}_model.pt", 'wb') as f:
            torch.save(model_state, f)

        

        loss_stats.append((ctr,epoch_loss/iter_ctr))
        acc.append((epoch,(validation_accuracy)))
        train_acc.append((epoch, (t_accuracy/data.train_instances)))
        test_acc.append((epoch, (test_accuracy)))

        with open(f"{log_dir}/{args['model_type']}_lossstats.pickle",'wb') as f:
            pickle.dump(loss_stats,f)
        with open(f"{log_dir}/{args['model_type']}_accstats.pickle",'wb') as f:
            pickle.dump(acc,f)
        
        with open(f"{log_dir}/{args['model_type']}_train_acc.pickle",'wb') as f:
            pickle.dump(train_acc,f)
        
        with open(f"{log_dir}/{args['model_type']}_test_acc.pickle", 'wb') as f:
            pickle.dump(test_acc, f)


# train_model(model, args['epochs'], data)

