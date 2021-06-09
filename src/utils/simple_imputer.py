import pandas as pd
import numpy as np

class SimpleImputer:
    def __init__(self):
        self.ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']

    def impute_mortality(self, df):
        idx = pd.IndexSlice
        df = df.copy()
        if len(df.columns.names) > 2: df.columns = df.columns.droplevel(('label', 'LEVEL1', 'LEVEL2'))
        
        df_out = df.loc[:, idx[:, ['mean', 'count']]]
        icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(self.ID_COLS).mean()
        
        df_out.loc[:,idx[:,'mean']] = df_out.loc[:,idx[:,'mean']].groupby(self.ID_COLS).fillna(
            method='ffill'
        ).groupby(self.ID_COLS).fillna(icustay_means).fillna(0)
        
        df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(float)
        df_out.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)
        
        is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
        hours_of_absence = is_absent.cumsum()
        time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].fillna(method='ffill')
        time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

        df_out = pd.concat((df_out, time_since_measured), axis=1)
        df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)
        
        df_out.sort_index(axis=1, inplace=True)
        return df_out
    
    def decay_imputer(self, df):
        '''
        given df: Nan will be imputed with zeros because we will create mask for Nan values
        time_since_measurement will be hourly sequence 
        '''
        idx = pd.IndexSlice
        df = df.copy()
        if len(df.columns.names) > 2: df.columns = df.columns.droplevel(('label', 'LEVEL1', 'LEVEL2'))
        
        df_out = df.loc[:, idx[:, ['mean', 'count']]]

        df_out.loc[:,idx[:,'mean']] = df_out.loc[:,idx[:,'mean']].groupby(self.ID_COLS).fillna(0)
        df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(float)
        df_out.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)

        is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
        hours_of_absence = is_absent.cumsum()
        time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].fillna(method='ffill')
        time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

        df_out = pd.concat((df_out, time_since_measured), axis=1)
        df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)
        
        df_out.sort_index(axis=1, inplace=True)
        return df_out