from pickle import TRUE
from numpy.core.defchararray import count
import torch
import numpy as np 
import logging
import json
from torch import nn as nn 
from src.utils.mimic_iii_decay_data import MIMICNonIidData
from src.model.mimic_lstm_model import MIMICLSTMModel
from src.model.mimic_gru_model import MIMICGRUModel
from src.model.mimic_decay_with_cb_loss import MIMICDecayCBLossModel
from src.model.mimic_decay_model import MIMICDecayModel
from src.model.mimic_grud_model import MIMICGRUDModel
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve
from src.utils.save_utils import MimicSave


np.random.seed(1048)
torch.manual_seed(1048)
torch.cuda.manual_seed(1048)

SAVE_DIR = 'mimic/non_iid_test'


class TestNonIid:
    def __init__(self, model_path, test_data_object):
        checkpoint = torch.load(model_path)
        args = checkpoint['args']
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        if args['model_type'] == 'RIMDecay'and args['is_cbloss']:
            self.model = MIMICDecayCBLossModel(args).to(self.device)
        elif args['model_type'] == 'RIMDecay' and not args['is_cbloss']:
            self.model = MIMICDecayModel(args).to(self.device)
        elif args['model_type'] == 'GRUD':
            self.model = MIMICGRUDModel(args).to(self.device)
        elif args['model_type'] == 'LSTM':
            self.model = MIMICLSTMModel(args).to(self.device)
        elif args['model_type'] == 'GRU':
            self.model = MIMICGRUModel(args).to(self.device)
        else:
            raise Exception('No model type found: {}'.format(model_path))
        
        self.data_object = test_data_object
        self.args = args
        self.set_model_name()
        self.model.load_state_dict(checkpoint['net'])
    
    def test_normal(self):
        all_data = self.get_all_non_iid_data()
        print(f"------- Initiating non-iid test with {self.args['model_type']} with {self.args['rnn_cell']}---------")
        stats = {}
        for hour,hour_data in all_data.items():
            print(f'--------- Testing @{hour}-hour data')
            x, static, x_mean, y = hour_data
            counts = torch.unique(y, return_counts=True)[1]
            # print(f'counts: [class 0]: {counts[0].item()}, [class 1]:{counts[1].item()}')
            static = static.to(self.device)
            x_mask = x[:, 1, :, :].to(self.device)
            delta = x[:, 2, :, :].to(self.device)
            x_mean = x_mean.to(self.device)
            x_last_ob = x[:, 3, :, :].to(self.device)
            x = x[:, 0, :, :].to(self.device)
            y = y.to(self.device)
            predictions = self.model(x, static, x_mask, delta, x_last_ob, x_mean)
            probs = torch.round(torch.sigmoid(predictions))
            
            y_truth = y.cpu().detach().numpy()
            y_pred = probs.view(-1).cpu().detach().numpy()

            report = classification_report(y_truth, y_pred, output_dict=True, zero_division=0)
            stats[f'{hour}'] = report
            print(report)

        del all_data
        return stats

    def test_cb_model(self):
        all_data = self.get_all_non_iid_data()
        print(f"------- Initiating non-iid test with {self.args['model_type']} with {self.args['rnn_cell']}, and CB-Loss custom function---------")
        stats = {}
        for hour,hour_data in all_data.items():
            print(f'--------- Testing @{hour}-hour data')
            x, static, x_mean, y = hour_data
            counts = torch.unique(y, return_counts=True)[1]
            # print(f'counts: [class 0]: {counts[0].item()}, [class 1]:{counts[1].item()}')
            static = static.to(self.device)
            x_mask = x[:, 1, :, :].to(self.device)
            delta = x[:, 2, :, :].to(self.device)
            x_mean = x_mean.to(self.device)
            x_last_ob = x[:, 3, :, :].to(self.device)
            x = x[:, 0, :, :].to(self.device)
            y = y.to(self.device)
            predictions = self.model(x, static, x_mask, delta, x_last_ob, x_mean)
            output = torch.argmax(predictions, dim=1)

            y_truth = y.cpu().detach().numpy()
            y_pred = output.view(-1).cpu().detach().numpy()

            report = classification_report(y_truth, y_pred, output_dict=True, zero_division=0)
            stats[f'{hour}'] = report
            print(report)
            stats[f'{hour}'] = report
        
        del all_data
        return stats

    def test_general_model(self):
        all_data = self.get_all_non_iid_data()
        print(f"------- Initiating non-iid test with {self.args['model_type']} with {self.args['rnn_cell']}---------")
        stats = {}
        for hour,hour_data in all_data.items():
            print(f'--------- Testing @{hour}-hour data')
            x, static, _, y = hour_data
            counts = torch.unique(y, return_counts=True)[1]
            # print(f'counts: [class 0]: {counts[0].item()}, [class 1]:{counts[1].item()}')
            static = static.to(self.device)
            x = x[:, 0, :, :].to(self.device)
            y = y.to(self.device)
            predictions = self.model(x, static)
            probs = torch.round(torch.sigmoid(predictions))
            
            y_truth = y.cpu().detach().numpy()
            y_pred = probs.view(-1).cpu().detach().numpy()

            report = classification_report(y_truth, y_pred, output_dict=True, zero_division=0)
            stats[f'{hour}'] = report
            print(report)
            stats[f'{hour}'] = report
        
        del all_data
        return stats

    def get_all_non_iid_data(self):
        test_18 = self.data_object.get_data(18)
        test_30 = self.data_object.get_data(30)
        test_36 = self.data_object.get_data(36)
        test_42 = self.data_object.get_data(42)
        test_48 = self.data_object.get_data(48)

        return {'18':test_18, '30':test_30, '36':test_36, '42':test_42, '48':test_48}

    def set_model_name(self):
        if self.args['model_type'] in ['LSTM', 'GRU']:
            self.model_name = self.args['model_type']
        else:
            self.model_name = f"{self.args['model_type']}_{self.args['rnn_cell']}"


# if __name__ =='__main__':
    # data_object = MIMICNonIidData('./data/mimic_iii/test_dump/non_iid_in_hospital.npz')
    # trainer = TestNonIid('./mimic/old_models/GRUD_GRU_model.pt', data_object)
    # results= trainer.test_general_model()
    # print("############ Result #####################")
    # print(results)

    # results= trainer.test_normal()
    # print("############ Result #####################")
    # print(results)
    
    if not (TRUE and False):
        print('-')
    else:
        print('+')
