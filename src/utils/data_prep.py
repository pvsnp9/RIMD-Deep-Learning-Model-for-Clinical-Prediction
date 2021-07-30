from pickle import NONE
import pandas as pd 
import numpy as np
import torch 
from .simple_imputer import SimpleImputer


class MortalityDataPrep:
    # data file path to read, parse, and impute
    def __init__(self, file_path, window_size = 24, gap_time = 6, type='in_hospital', non_iid = [18,30,36,42,48]):
        self.file_path = file_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gap_time = gap_time
        self.window_size = window_size
        self.ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
        self.type = type
        self.non_iid = non_iid
        self.imputer = SimpleImputer()


    def read_file(self):
        #nned to look for tables package
        all_data = pd.read_pickle(self.file_path)
        self.vital_labs = all_data['vitals_labs']
        self.statics = all_data['patients']
        del all_data
    
    def prepare_non_iid(self, destination_dir, file_path=None):

        self.read_file()
        print('Preparing non_iid data')
        if file_path is not None:
            self.file_path = file_path

        hourly_data = {}
        for hour in self.non_iid:
            targets = self.statics[self.statics.max_hours > hour + self.gap_time][['mort_hosp', 'mort_icu','los_icu']]
            targets.astype(float)

            '''
            extract temporal data for each patients i.e. targets  inner join with icu_stay.
            a patient can have multiple icu_stay but we have picked first [mimic_extract]
            '''
            vital_labs = self.vital_labs[(self.vital_labs.index.get_level_values('icustay_id').isin(set(targets.index.get_level_values('icustay_id')))) & (self.vital_labs.index.get_level_values('hours_in') < hour)]
            statics = self.statics[(self.statics.index.get_level_values('icustay_id').isin(set(targets.index.get_level_values('icustay_id'))))]

            assert set(vital_labs.index.get_level_values('subject_id')) == set(targets.index.get_level_values('subject_id')), "Data and target Subject ID pools differ !!"
            assert set(statics.index.get_level_values('subject_id')) == set(set(vital_labs.index.get_level_values('subject_id'))), "Data and static variable subject ID pool differs !!"

            statics = self.static_variables(['gender', 'age', 'ethnicity', 'first_careunit'], statics)

            # Normalization
            idx = pd.IndexSlice
            means, stdv = vital_labs.loc[:, idx[:, 'mean']].mean(axis=0), vital_labs.loc[:, idx[:, 'mean']].std(axis=0)

            vital_labs.loc[:, idx[:, 'mean']] = (vital_labs.loc[:, idx[:, 'mean']] - means) / stdv
            

            vital_labs = self.imputer.decay_imputer(vital_labs)
            statics.isnull().any().any(), 'Null Found in static features'
            vital_labs.isnull().any().any(), 'Null found in variable feature'
            
            '''
            compute icu_stay mean for each patients
            '''
            icu_stay_mean = vital_labs.groupby(self.ID_COLS).mean()
            icu_stay_mean = icu_stay_mean.loc[:, idx[:, 'mean']]
            x_mean = np.expand_dims(icu_stay_mean, axis=1)
            
            del icu_stay_mean


            if self.type == 'in_icu':
                targets.drop(columns=['mort_hosp', 'los_icu'], inplace=True)
            elif self.type == 'los_icu':
                targets.drop(columns=['mort_icu', 'mort_hosp'], inplace=True)
            else:
                targets.drop(columns=['mort_icu', 'los_icu'], inplace=True)

            variable_data = vital_labs.loc[:, idx[:, 'mean']]
            mask = vital_labs.loc[:,pd.IndexSlice[:, 'mask']].to_numpy()
            delta = vital_labs.loc[:,pd.IndexSlice[:, 'time_since_measured']].to_numpy()

            missing_idxs = np.where(mask == 0)

            variable_data = variable_data.droplevel('Aggregation Function', axis=1)
            x = variable_data.to_numpy()
            s = statics.to_numpy()
            y = targets.to_numpy()
            variable_last_observed = np.copy(x)
            only_last_observed = [variable_last_observed[i-1,j] if i != 0 else variable_last_observed[i,j] for i,j in zip(missing_idxs[0], missing_idxs[1])]
            variable_last_observed[missing_idxs] = only_last_observed

            hourly_data[hour] = [x,y,s,x_mean, mask, delta,variable_last_observed] 
            # {
            #     'x': x, 'y':y, 'statics': s, 'xmean':x_mean, 'mask':mask,
            #     'delta':delta, 'lastob': variable_last_observed
            # }
        order = ['x', 'y', 'statics','xmean', 'mask', 'delta', 'lastob']
        np.savez_compressed(f'{destination_dir}/non_iid_{self.type}.npz', 
                                order = order,
                                h_18 = hourly_data[18],
                                h_30 = hourly_data[30],
                                h_36 = hourly_data[36],
                                h_42 = hourly_data[42],
                                h_48 = hourly_data[48]
                            )
                            
        saved_file = f'{destination_dir}/non_iid_{self.type}.npz'
        return saved_file
    
    def preprocess_decay(self, save_npy = True, destination_dir=''):
        self.read_file()

        # pick targets who has max_hours >= 24 hrs
        targets = self.statics[self.statics.max_hours > self.window_size + self.gap_time][['mort_hosp', 'mort_icu', 'los_icu']]
        targets.astype(float)

        '''
        extract temporal data for each patients i.e. targets  inner join with icu_stay.
        a patient can have multiple icu_stay but we have picked first [mimic_extract]
        '''
        self.vital_labs = self.vital_labs[(self.vital_labs.index.get_level_values('icustay_id').isin(set(targets.index.get_level_values('icustay_id')))) & (self.vital_labs.index.get_level_values('hours_in') < self.window_size)]
        self.statics = self.statics[(self.statics.index.get_level_values('icustay_id').isin(set(targets.index.get_level_values('icustay_id'))))]

        assert set(self.vital_labs.index.get_level_values('subject_id')) == set(targets.index.get_level_values('subject_id')), "Data and target Subject ID pools differ !!"
        assert set(self.statics.index.get_level_values('subject_id')) == set(set(self.vital_labs.index.get_level_values('subject_id'))), "Data and static variable subject ID pool differs !!"

        self.statics = self.static_variables(['gender', 'age', 'ethnicity', 'first_careunit'])

        # Normalization
        idx = pd.IndexSlice
        means, stdv = self.vital_labs.loc[:, idx[:, 'mean']].mean(axis=0), self.vital_labs.loc[:, idx[:, 'mean']].std(axis=0)

        self.vital_labs.loc[:, idx[:, 'mean']] = (self.vital_labs.loc[:, idx[:, 'mean']] - means) / stdv
        
        

        self.vital_labs = self.imputer.decay_imputer(self.vital_labs)
        self.statics.isnull().any().any(), 'Null Found in static features'
        self.vital_labs.isnull().any().any(), 'Null found in variable feature'

        '''
        compute icu_stay mean for each patients [24 hrs]
        '''
        icu_stay_mean = self.vital_labs.groupby(self.ID_COLS).mean()
        icu_stay_mean = icu_stay_mean.loc[:, idx[:, 'mean']]
        x_mean = np.expand_dims(icu_stay_mean, axis=1)
        
        del icu_stay_mean
        

        if self.type == 'in_icu':
            targets.drop(columns=['mort_hosp', 'los_icu'], inplace=True)
        elif self.type == 'los_icu':
            targets.drop(columns=['mort_icu', 'mort_hosp'], inplace=True)
        else:
            targets.drop(columns=['mort_icu', 'los_icu'], inplace=True)

        if save_npy:
            variable_data = self.vital_labs.loc[:, idx[:, 'mean']]
            mask = self.vital_labs.loc[:,pd.IndexSlice[:, 'mask']].to_numpy()
            delta = self.vital_labs.loc[:,pd.IndexSlice[:, 'time_since_measured']].to_numpy()

            missing_idxs = np.where(mask == 0)

            variable_data = variable_data.droplevel('Aggregation Function', axis=1)
            x = variable_data.to_numpy()
            s = self.statics.to_numpy()
            y = targets.to_numpy()
            variable_last_observed = np.copy(x)
            only_last_observed = [variable_last_observed[i-1,j] if i != 0 else variable_last_observed[i,j] for i,j in zip(missing_idxs[0], missing_idxs[1])]
            variable_last_observed[missing_idxs] = only_last_observed

            print(f'Saving numpy files to {destination_dir}')
            print(f'[Main data]: Variable feature: {x.shape} static feature: {s.shape}, target: {y.shape}')
            print(f'[Additional data]: x_mean:{x_mean.shape}, mask:{mask.shape}, delta:{delta.shape}, last Ob:{variable_last_observed.shape} ')
            
            np.savez_compressed(f'{destination_dir}/decay_x_y_statics_xmean_mask_delta_lastob_{self.type}_{targets.shape[0]}.npz', x= x, y=y, statics = s, xmean=x_mean, mask=mask, delta=delta,lastob=variable_last_observed)
            # with open(f'{destination_dir}/all_data_{targets.shape[0]}_npy.pkl', 'wb') as handle:
            #     pickle.dump({'x':x, 'statics':s, 'y':y}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # statics = statics.reset_index()
        # targets = targets.reset_index()
        saved_file = f'{destination_dir}/decay_x_y_statics_xmean_mask_delta_lastob_{self.type}_{targets.shape[0]}.npz'
        return saved_file
    
    def preprocess(self, save_npy = True, destination_dir=''):
        self.read_file()

        targets = self.statics[self.statics.max_hours > self.window_size + self.gap_time][['mort_hosp', 'mort_icu', 'los_icu']]
        targets.astype(float) #index (ID_COLS) and data cols ['mort_hosp', 'mort_icu']
        self.vital_labs = self.vital_labs[(self.vital_labs.index.get_level_values('icustay_id').isin(set(targets.index.get_level_values('icustay_id')))) & (self.vital_labs.index.get_level_values('hours_in') < self.window_size)]
        self.statics = self.statics[(self.statics.index.get_level_values('icustay_id').isin(set(targets.index.get_level_values('icustay_id'))))]

        assert set(self.vital_labs.index.get_level_values('subject_id')) == set(targets.index.get_level_values('subject_id')), "Data and target Subject ID pools differ !!"
        assert set(self.statics.index.get_level_values('subject_id')) == set(set(self.vital_labs.index.get_level_values('subject_id'))), "Data and static variable subject ID pool differs !!"

        self.statics = self.static_variables(['gender', 'age', 'ethnicity', 'first_careunit'])

        # Normalization
        idx = pd.IndexSlice
        means, stdv = self.vital_labs.loc[:, idx[:, 'mean']].mean(axis=0), self.vital_labs.loc[:, idx[:, 'mean']].std(axis=0)

        self.vital_labs.loc[:, idx[:, 'mean']] = (self.vital_labs.loc[:, idx[:, 'mean']] - means) / stdv
        # imputation
        self.vital_labs = self.imputer.impute_mortality(self.vital_labs)
        self.statics.isnull().any().any(), 'Null Found in static features'
        self.vital_labs.isnull().any().any(), 'Null found in variable feature'
        

        if self.type == 'in_icu':
            targets.drop(columns=['mort_hosp', 'los_icu'], inplace=True)
        elif self.type == 'los_icu':
            targets.drop(columns=['mort_icu', 'mort_hosp'], inplace=True)
        else:
            targets.drop(columns=['mort_icu', 'los_icu'], inplace=True)

        if save_npy:
            variable_mean = self.vital_labs.loc[:, idx[:, 'mean']]
            variable_mean = variable_mean.droplevel('Aggregation Function', axis=1)
            x = variable_mean.to_numpy()
            s = self.statics.to_numpy()
            y = targets.to_numpy()

            print(f'Saving numpy files to {destination_dir}')
            print(f'sizes: \n Variable feature: {x.shape} static feature: {s.shape}, target: {y.shape}')
            
            np.savez_compressed(f'{destination_dir}/x_y_statics_{self.type}_{targets.shape[0]}.npz', x= x, y=y, statics = s)
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

    def static_variables(self, variables, static_data =None):
        try:
            if static_data is not None:
                statics_data = static_data[variables]
            else:
                statics_data = self.statics[variables]
            # statics_data.loc[:, 'intime'] = statics_data['intime'].astype('datetime64').apply(lambda x: x.hour)
            statics_data.loc[:, 'age'] = statics_data['age'].apply(self.categorize_age)
            statics_data.loc[:, 'ethnicity'] = statics_data['ethnicity'].apply(self.categorize_ethnicity)
            statics_data = pd.get_dummies(statics_data, columns= ['gender', 'age', 'ethnicity', 'first_careunit'])

            return statics_data
        except KeyError:
            print('Given variables does not exist in statics data frame !!')

        
#if __name__ == "__main__":
 #   d = MortalityDataPrep('data/mimic_iii/test_dump/all_hourly_data_10hr.pkl', type='los_icu')
  #  print('Reading and prepprocessing data .....')
   # x = d.preprocess_decay(destination_dir='data/mimic_iii/test_dump')
    # x = d.prepare_non_iid('data/mimic_iii/test_dump','data/mimic_iii/test_dump/all_hourly_data.pkl' )
    #print(f'Preprocessing Done, and saved .npz files @ {x}')