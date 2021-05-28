import pandas as pd 
import numpy as np
import torch 
from simple_imputer import SimpleImputer

class MortalityDataPrep:
    # data file path to read, parse, and impute
    def __init__(self, file_path, window_size = 24, gap_time = 6, type='in_hospital'):
        self.file_path = file_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gap_time = gap_time
        self.window_size = window_size
        self.ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
        self.type = type
        self.imputer = SimpleImputer()

    def read_file(self):
        #nned to look for tables package
        all_data = pd.read_pickle(self.file_path)
        self.vital_labs = all_data['vitals_labs']
        self.statics = all_data['patients']
        del all_data
    
    def preprocess(self, save_npy = True, destination_dir=''):
        self.read_file()

        targets = self.statics[self.statics.max_hours > self.window_size + self.gap_time][['mort_hosp', 'mort_icu']]
        targets.astype(float) #index (ID_COLS) and data cols ['mort_hosp', 'mort_icu']
        self.vital_labs = self.vital_labs[(self.vital_labs.index.get_level_values('icustay_id').isin(set(targets.index.get_level_values('icustay_id')))) & (self.vital_labs.index.get_level_values('hours_in') < self.window_size)]
        self.statics = self.statics[(self.statics.index.get_level_values('icustay_id').isin(set(targets.index.get_level_values('icustay_id'))))]

        assert set(self.vital_labs.index.get_level_values('subject_id')) == set(targets.index.get_level_values('subject_id')), "Data and target Subject ID pools differ !!"
        assert set(self.statics.index.get_level_values('subject_id')) == set(set(self.vital_labs.index.get_level_values('subject_id'))), "Data and static variable subject ID pool differs !!"

        self.statics = self.static_variables(['gender', 'age', 'ethnicity', 'first_careunit', 'intime'])

        # Normalization
        idx = pd.IndexSlice
        means, stdv = self.vital_labs.loc[:, idx[:, 'mean']].mean(axis=0), self.vital_labs.loc[:, idx[:, 'mean']].std(axis=0)

        self.vital_labs.loc[:, idx[:, 'mean']] = (self.vital_labs.loc[:, idx[:, 'mean']] - means) / stdv
        # imputation
        self.vital_labs = self.imputer.impute_mortality(self.vital_labs)
        self.statics.isnull().any().any(), 'Null Found in static features'
        self.vital_labs.isnull().any().any(), 'Null found in variable feature'
        

        if self.type == 'in_icu':
            targets.drop(columns=['mort_hosp'], inplace=True)
        else:
            targets.drop(columns=['mort_icu'], inplace=True)

        if save_npy:
            variable_mean = self.vital_labs.loc[:, idx[:, 'mean']]
            variable_mean = variable_mean.droplevel('Aggregation Function', axis=1)
            x = variable_mean.to_numpy()
            s = self.statics.to_numpy()
            y = targets.to_numpy()

            print(f'Saving numpy files to {destination_dir}')
            print(f'sizes: \n Variable feature: {x.shape} static feature: {s.shape}, target: {y.shape}')
            
            np.savez_compressed(f'{destination_dir}/x_y_statics_{targets.shape[0]}.npz', x= x, y=y, statics = s)
            # with open(f'{destination_dir}/all_data_{targets.shape[0]}_npy.pkl', 'wb') as handle:
            #     pickle.dump({'x':x, 'statics':s, 'y':y}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # statics = statics.reset_index()
        # targets = targets.reset_index()
        saved_file = f'{destination_dir}/x_y_statics_{targets.shape[0]}.npz'
        return self.vital_labs, targets, self.statics, saved_file


    def categorize_age(self, age):
        if age > 10 and age <= 30:
            cat = 1
        elif age > 30 and age <= 50:
            cat = 2
        elif age > 50 and age <= 70:
            cat = 3
        else:
            cat = 2
        
        return cat

    def categorize_ethnicity(self, ethnicity):
        if 'AMERICAN INDIAN' in ethnicity:
            ethnicity = 'AMERICAN INDIAN'
        elif 'ASIAN' in ethnicity:
            ethnicity = 'ASIAN'
        elif 'WHITE' in ethnicity:
            ethnicity = 'WHITE'
        elif 'HISPANIC' in ethnicity:
            ethnicity = 'HISPANIC/LATINO'
        elif 'BLACK' in ethnicity:
            ethnicity = 'BLACK'
        else: 
            ethnicity = 'OTHER'
        return ethnicity

    def static_variables(self, variables):
        try:
            statics = self.statics[variables]
            statics.loc[:, 'intime'] = statics['intime'].astype('datetime64').apply(lambda x: x.hour)
            statics.loc[:, 'age'] = statics['age'].apply(self.categorize_age)
            statics.loc[:, 'ethnicity'] = statics['ethnicity'].apply(self.categorize_ethnicity)
            statics = pd.get_dummies(statics, columns= ['gender', 'age', 'ethnicity', 'first_careunit'])

            return statics
        except KeyError:
            print('Given variables does not exist in statics data frame !!')

    def to_3d_array(self,data):
        # subject = set(data.)
        pass
        
# if __name__ == "__main__":
#     d = MortalityDataPrep('data/mimic_iii/curated_30k/all_hourly_data_30000.pkl')
#     print('Reading and prepprocessing data .....')
#     x = d.preprocess(True,  'data/mimic_iii/preprocessed/mortality_and_los')
#     print('Preprocessing Done, and saved .npz files')