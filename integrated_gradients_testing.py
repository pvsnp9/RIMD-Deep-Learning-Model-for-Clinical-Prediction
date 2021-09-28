import pickle
from captum import attr
import torch, gc
import numpy as np 
from src.model.mimic_decay_with_cb_loss import MIMICDecayCBLossModel
from src.utils.mimic_iii_decay_data import MIMICNonIidData
from src.utils.data_prep import MortalityDataPrep
import matplotlib.pyplot as plt 
from captum.attr import IntegratedGradients
import pandas as pd
import pickle

gc.collect()
torch.cuda.empty_cache()

class IntegratedGradientsCheck():
    def __init__(self, model_path, data_object):
        checkpoint = torch.load(model_path)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        args = checkpoint['args']
        self.model = MIMICDecayCBLossModel(args).to(self.device)
        self.model.load_state_dict(checkpoint['net'])
        self.data_object = data_object
        

    
    def run_integrated_gradients(self, hour = 30):
        # feature_list = self.get_feature_list()
        # print(f'Feature list: {feature_list}')
        x, statics, x_mean, _ = self.get_data(hour)
        print(f"Model Description: \n {self.model}")
        ig = IntegratedGradients(self.model)
        
        x_mask = x[:, 1, :, :].requires_grad_().to(self.device)
        delta = x[:, 2, :, :].requires_grad_().to(self.device)
        x_last_ob = x[:, 3, :, :].requires_grad_().to(self.device)
        x = x[:, 0, :, :].requires_grad_().to(self.device)
        statics = statics.float().requires_grad_().to(self.device)
        x_mean = x_mean.requires_grad_().to(self.device)

        attr, delta = ig.attribute(x, additional_forward_args=(statics, x_mask, delta, x_last_ob, x_mean),target=1, return_convergence_delta=True)
        attr = attr.cpu().detach().numpy()
        delta = delta.cpu().detach().numpy()

        return attr, delta


    def get_feature_list(self, file_path='./data/mimic_iii/test_dump/all_hourly_data.pkl'):
        print(f"reading feature list from file: {file_path}")
        all_data = pd.read_pickle(file_path)
        vital_labs = all_data['vitals_labs']
        statics = all_data['patients']
        del all_data

        md = MortalityDataPrep(file_path)
        main_static_feat = ['gender', 'age', 'ethnicity', 'first_careunit']
        static_feat = md.static_variables(main_static_feat, statics)
        del statics
        static_features = static_feat.columns.tolist()

        #temporal feature
        temporal_feat_list = vital_labs.columns.tolist()
        del vital_labs
        temporal_features = sorted(set([x for x,y in temporal_feat_list]))

        return temporal_features, static_features, [*temporal_features, * static_features]



    def get_data(self, hour):
        return self.data_object.get_data(hour)

if __name__ == "__main__":
    saved_model_directory = './test_models/model'
    save_directory = './test_models/ig_testing'
    # we are going to perform integrated testing only for our bets models [RIMDecay_LSTM, RIMDecay_GRU]
    data_object = MIMICNonIidData('./data/mimic_iii/test_dump/non_iid_in_hospital.npz', 100)
    models = ['RIMDecay_LSTM', 'RIMDecay_GRU']

    ig_data = {}
    for model in models:
        ig_test_object = IntegratedGradientsCheck(f'{saved_model_directory}/{model}_cbloss_model.pt', data_object)
        attr, delta = ig_test_object.run_integrated_gradients()
        del ig_test_object
        ig_data[f'{model}_cbloss_model'] = {
            'attr': attr, 
            'delta': delta
        }

    with open(f'{save_directory}/ig_testing.pickle', 'wb') as handle:
        pickle.dump(ig_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Gradient extract and saved ad {save_directory}')

    # data checking
    # with open(f'{save_directory}/ig_testing.pickle', 'rb') as handle:
    #     ig_grad_data = pickle.load(handle)

    # print(ig_grad_data['RIMDecay_LSTM_cbloss_model'])
    # ig_grad_data = np.load(f'{save_directory}/ig_testing.npz', allow_pickle=True)
    # data = ig_grad_data['RIMDecay_LSTM_cbloss_model']
    # print(data['attr'])