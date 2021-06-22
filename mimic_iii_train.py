import torch
import pickle
import numpy as np
from src.model.mimic_model import MIMICModel
from src.model.mimic_lstm_model import MIMICLSTMModel
from src.model.mimic_gru_model import MIMICGRUModel
from src.utils.mimic_iii_data import MIMICIIIData
from src.utils.data_prep import MortalityDataPrep
from src.model.mimic_decay_model import MIMICDecayModel
from src.utils.mimic_iii_decay_data import MIMICDecayData
from src.utils.classification_report import ClassificationReport
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve

'''
!!!! Important !!!!!!!
Input arguments like model_type and cell type are crucial.
'''
args = {
    'epochs':100,
    'batch_size': 128,
    'input_size': 1, #automatically picked from data
    'model_type': 'RNN', # type of model  RIM, LSTM, GRU
    'hidden_size': 100,
    'num_rims': 4,
    'lr': 0.0001,
    'rnn_cell': 'GRU_D', # type of cell LSTM, or GRU
    'input_key_size': 64,
    'input_value_size': 128,
    'input_query_size': 64,
    'num_input_heads': 2,
    'input_dropout': 0.2,
    'comm_key_size': 64,
    'comm_value_size': 100,
    'comm_query_size': 64,
    'num_comm_heads': 2,
    'comm_dropout': 0.2,
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

# torch.manual_seed(10)
# np.random.seed(10)
# torch.cuda.manual_seed(10)


save_dir = 'mimic/models'
log_dir = 'mimic/logs'

# Data preprocessing
if(args['need_data_preprocessing']):
    prep_data = MortalityDataPrep(args['raw_data_file_path'])
    if args['mask']:
        args['input_file_path'] = prep_data.preprocess_decay(True, args['processed_data_path'])
    else:
        _, _, _, args['input_file_path'] = prep_data.preprocess(True, args['processed_data_path'])
        del _

class TrainModels:
    def __init__(self, args=None):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if args is not None:
            self.args = args
            self.data_object = None
            self.model = None
            self.cell_types = ['LSTM', 'GRU']
            if args['model_type'] == 'RIMDecay':
                self.set_decay_params()
            else:
                self.set_default_params()
        else:
            print('-------------- Model initiated for prediction ---------------------')

    def set_decay_params(self):
        self.data_object = MIMICDecayData(args['batch_size'], 24, args['input_file_path'])
        self.args['input_size'] = self.data_object.input_size
        self.args['static_features'] = self.data_object.statics_size
        self.args['mask_size'] = self.data_object.input_size
        self.args['delta_size'] = self.data_object.input_size
        self.args['hidden_size'] = self.data_object.input_size
        self.args['comm_value_size'] = self.data_object.input_size

    def set_default_params(self):
        self.data_object = MIMICIIIData(args['batch_size'], 24, args['input_file_path'], args['mask'])
        self.args['input_size'] = self.data_object.input_size
        self.args['static_features'] = self.data_object.statics_size
        if self.args['model_type'] == 'LSTM': self.cell_types.pop(0)
        elif self.args['model_type'] == 'GRU': self.cell_types.pop(1)

    def train(self):
        acc = []
        train_acc = []
        test_acc = []
        loss_stats = []
        ctr = 0
        start_epochs = 0

        max_val_accuracy = 0.0
        epochs_no_improve = 0
        max_no_improvement = 5
        improvement_threshold = 0.0001 

        for cell in self.cell_types:
            train_loader, val_loader, test_loader = self.data_object.data_loader()
            self.args['rnn_cell'] = cell
            if self.args['model_type'] == 'RIMDecay':
                self.model = MIMICDecayModel(self.args).to(self.device)
            elif self.args['model_type'] == 'RIM':
                self.model = MIMICModel(self.args).to(self.device)
            elif self.args['model_type'] == 'LSTM':
                self.model = MIMICLSTMModel(self.args).to(self.device)
            else:
                self.model = MIMICGRUModel(self.args).to(self.device)
            
            print(f'Model Arch: \n {self.model}')    
            print(f"Training, Validating, and Testing: {args['model_type']} model with {args['rnn_cell']} cell ")

            optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args['lr'])
            for epoch in range(start_epochs, self.args['epochs']):
                print(f' EPOCH: {epoch +1}')
                epoch_loss = 0.0
                iter_ctr = 0.0
                t_accuracy = 0
                norm = 0

                self.model.train()
                if self.args['model_type'] == 'RIMDecay':
                    for x, static, x_mean, y in train_loader:
                        iter_ctr += 1

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
                else:
                    for x, statics, y in train_loader:
                        iter_ctr += 1

                        x = x.to(self.device)
                        statics = statics.to(self.device)
                        y = y.to(self.device)

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

            
                validation_accuracy = self.eval(val_loader)
                test_accuracy = self.eval(test_loader)
                print(f'epoch loss: {epoch_loss}, taining accuracy: {t_accuracy/self.data_object.train_instances}, validation accuracy: {validation_accuracy}, Test accuracy: {test_accuracy}')
                
                # early stopping code
                improve = validation_accuracy - max_val_accuracy
                if improve > improvement_threshold:
                    epochs_no_improve = 0
                    max_val_accuracy = validation_accuracy

                    best_model_state = {
                    'net': self.model.state_dict(),
                    'epochs': epoch,
                    'args':args
                    }
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve == max_no_improvement:
                    print(f"Early Stopping @ EPOCH:[{epoch}]")
                    break

            print("saving the models state...")
            with open(f"{save_dir}/{args['model_type']}_{args['rnn_cell']}_model.pt", 'wb') as f:
                torch.save(best_model_state, f)

            loss_stats.append((ctr,epoch_loss/iter_ctr))
            acc.append((epoch,(validation_accuracy)))
            train_acc.append((epoch, (t_accuracy/self.data_object.train_instances)))
            test_acc.append((epoch, (test_accuracy)))

            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_lossstats.pickle",'wb') as f:
                pickle.dump(loss_stats,f)
            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_accstats.pickle",'wb') as f:
                pickle.dump(acc,f)
            
            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_train_acc.pickle",'wb') as f:
                pickle.dump(train_acc,f)
            
            with open(f"{log_dir}/{args['model_type']}_{args['rnn_cell']}_test_acc.pickle", 'wb') as f:
                pickle.dump(test_acc, f)

    def eval(self, data_loader):
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            if self.args['model_type'] == 'RIMDecay':
                for x, static, x_mean, y in data_loader:
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
            else:
                for x, statics, y in data_loader:
                    x = x.to(self.device)
                    statics = statics.to(self.device)
                    y = y.to(self.device)

                    predictions = self.model(x, statics)

                    probs = torch.round(torch.sigmoid(predictions))
                    correct = probs.view(-1) == y
                    accuracy += correct.sum().item()

        # todo compute accuracy
        accuracy /= self.data_object.dev_instances
        return accuracy

    def test(self, model_path, test_data):
        checkpoint = torch.load(model_path)
        args = checkpoint['args']
        if args['model_type'] == 'RIMDecay':
            model = MIMICDecayModel(args).to(self.device)
        elif args['model_type'] == 'RIM':
            model = MIMICModel(args).to(self.device)
        elif args['model_type'] == 'LSTM':
            model = MIMICLSTMModel(args).to(self.device)
        elif args['model_type'] == 'GRU':
            model = MIMICGRUModel(args).to(self.device)
        else:
            raise Exception('No model type found')

        model.load_state_dict(checkpoint['net'])
        print(f'Loaded model arch: \n {model}')

        if args['model_type'] == 'RIMDecay':
            x, static, x_mean, y = test_data
            static = static.to(self.device)
            x_mask = x[:,1,:,:].to(self.device)
            delta = x[:,2,:,:].to(self.device)
            x_mean = x_mean.to(self.device)
            x_last_ob = x[:,3,:,:].to(self.device)
            x = x[:,0,:,:].to(self.device)
            y = y.to(self.device)
            predictions = model(x, static, x_mask, delta, x_last_ob, x_mean)
            probs = torch.round(torch.sigmoid(predictions))

        else:
            x, statics, y = test_data
            x = x.to(self.device)
            statics = statics.to(self.device)
            y = y.to(self.device)
            predictions = model(x, statics)
            probs = torch.round(torch.sigmoid(predictions))
        
        gt = y.cpu().detach().numpy()
        pt = probs.view(-1).cpu().detach().numpy()
        y_score = predictions.view(-1).cpu().detach().numpy()

        return gt, pt, y_score



''' Training '''
model_type = ['RIMDecay','RIM', 'LSTM', 'GRU']
cell_type= ['LSTM', 'GRU']
for model in model_type:
    args['model_type'] = model
    model_trainer = TrainModels(args)
    model_trainer.train()
        


# def make_report():
#     data  = MIMICIIIData(64, 24, args['input_file_path'])  #MIMICDecayData(64, 24, args['input_file_path'])
#     test_data = data.get_test_data()
#     model_path = f'./mimic/models/RIM_LSTM_model.pt'
#     trainer = TrainModels()
#     y_truth, y_pred, y_score = trainer.test(model_path, test_data)

#     clf = ClassificationReport(None, y_truth, y_pred, y_score, 'RIM - LSTM','./figures')

#     results = clf.get_all_metrics()
#     cf_m = clf.get_confusion_matrix()
#     print(results)
#     print(cf_m)
#     print(classification_report(y_truth, y_pred))


# make_report()


# train_model(model, args['epochs'], data)

