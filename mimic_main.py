

"""
!!!! Important !!!!!!!
Input arguments like model_type and cell type are crucial.
"""
import pandas as pd

from mimic_iii_train import TrainModels
from src.model.mimic_ml_models import MimicMlTrain
from src.utils.data_prep import MortalityDataPrep
from src.utils.mimic_evaluation import MIMICReport
from src.utils.mimic_iii_data import MIMICIIIData
from src.utils.mimic_iii_decay_data import MIMICDecayData

OUTPUT = 'output/plots'
N_HYPER_PARAM_SET = 1

save_dir = 'mimic/models'
log_dir = 'mimic/logs'

args = {
    'epochs':10,
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
    'input_file_path':'data/mimic_iii/test_dump/x_y_statics_20926.npz',
    'decay_input_file_path':'data/mimic_iii/test_dump/decay_data_20926.npz'
}

def mimic_main(run_type):
    # Data preprocessing
    if (args['need_data_preprocessing']):
        prep_data = MortalityDataPrep(args['raw_data_file_path'])
        if args['mask']:
            args['input_file_path'] = prep_data.preprocess_decay(True, args['processed_data_path'])
        else:
            _, _, _, args['input_file_path'] = prep_data.preprocess(True, args['processed_data_path'])
            del _

    #load datasets
    decay_data_object = MIMICDecayData(args['batch_size'], 24, args['decay_input_file_path'])
    data_object = MIMICIIIData(args['batch_size'], 24, args['input_file_path'], args['mask'])
    model_reports = {}
    if run_type == "train":
        #ML models first
        ml_trainer = MimicMlTrain(data_object, './mimic/models', OUTPUT, N_HYPER_PARAM_SET)
        ml_trainer.run()

        model_reports.update(ml_trainer.get_reports())

        #DL Models
        model_type = [  'RIMDecay','LSTM', 'GRU','RIM']
        cell_type = ['LSTM', 'GRU']
        for model in model_type:
            args['model_type'] = model
            if args['model_type'] == 'RIMDecay':
                dl_trainer = TrainModels(args,decay_data_object)
            else:
                dl_trainer = TrainModels(args,data_object)

            train_res =  dl_trainer.train()
            for model, res_sets in train_res.items():
                y_truth, y_pred, y_score = res_sets
                report = MIMICReport(model, y_truth, y_pred, y_score, './figures')
                model_reports.update({model:report})
    else:
        #ML Test
        ml_trainer = MimicMlTrain(data_object, './mimic/models', OUTPUT)
        model_reports.update(ml_trainer.test())

        #DL Test
        model_type = ['RIM_GRU', 'RIMDecay_GRU','RIM_LSTM', 'RIMDecay_LSTM','LSTM_GRU', 'GRU_LSTM']
        cell_type = ['LSTM', 'GRU']

        for model in model_type:
            if model.startswith('RIMDecay'):
                test_data = decay_data_object.get_test_data()
            else:
                test_data = data_object.get_test_data()
            trainer = TrainModels()
            model_path = f"{save_dir}/{model}_model.pt"
            y_truth, y_pred, y_score = trainer.test(model_path, test_data)
            report = MIMICReport(model, y_truth, y_pred, y_score, './figures')
            model_reports.update({model:report})

    results = {}
    cms = {}
    for model, report in model_reports.items():
        results.update(report.get_all_metrics())
        cms.update({model:report.get_confusion_matrix()})

    df_results =  pd.DataFrame(results)
    df_results = df_results.T
    print(df_results.to_markdown())

mimic_main("train")