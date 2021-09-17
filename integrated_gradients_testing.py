import torch, gc
import numpy as np 
from src.model.mimic_decay_with_cb_loss import MIMICDecayCBLossModel
from src.utils.mimic_iii_decay_data import MIMICNonIidData
from src.utils.data_prep import MortalityDataPrep
import matplotlib.pyplot as plt 
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
from scipy import stats
import pandas as pd

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
        

    
    def run_integrated_gradients(self, hour = 18):
        # feature_list = self.get_feature_list()
        # print(f'Feature list: {feature_list}')
        x, statics, x_mean, _ = self.get_data(hour)
        print(x.shape)
        print(f"Model Description: \n {self.model}")
        ig = IntegratedGradients(self.model)
        
        x_mask = x[:, 1, :, :].requires_grad_().to(self.device)
        delta = x[:, 2, :, :].requires_grad_().to(self.device)
        x_last_ob = x[:, 3, :, :].requires_grad_().to(self.device)
        x = x[:, 0, :, :].requires_grad_().to(self.device)
        statics = statics.float().requires_grad_().to(self.device)
        x_mean = x_mean.requires_grad_().to(self.device)

        attr, delta = ig.attribute(x, additional_forward_args=(statics, x_mask, delta, x_last_ob, x_mean),target=1, return_convergence_delta=True)
        # attr = attr.detach().numpy()
        x_imp = attr[0]
        print(x_imp)
        print(len(attr))


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
        static_feat = static_feat.columns.tolist()

        #temporal feature
        temporal_feat_list = vital_labs.columns.tolist()
        del vital_labs
        all_feature_list = sorted(set([x for x,y in temporal_feat_list]))

        all_feature_list.extend(static_feat)
        return all_feature_list



    def get_data(self, hour):
        return self.data_object.get_data(hour)

if __name__ == "__main__":
    data_object = MIMICNonIidData('./data/mimic_iii/non_iid_in_hospital.npz')
    ig_test_object = IntegratedGradientsCheck('./mimic/old_models/RIMDecay_LSTM_cbloss_model.pt', data_object)
    ig_test_object.run_integrated_gradients()