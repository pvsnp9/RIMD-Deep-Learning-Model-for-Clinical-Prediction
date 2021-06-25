

"""
!!!! Important !!!!!!!
Input arguments like model_type and cell type are crucial.
"""
import pandas as pd
import pickle
from mimic_iii_train import TrainModels
from src.model.mimic_ml_models import MimicMlTrain
from src.utils.data_prep import MortalityDataPrep
from src.utils.mimic_evaluation import MIMICReport
from src.utils.mimic_iii_data import MIMICIIIData
from src.utils.mimic_iii_decay_data import MIMICDecayData
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


OUTPUT = 'output/plots'
N_HYPER_PARAM_SET = 1

save_dir = 'mimic/models'
log_dir = 'mimic/logs'

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
    'input_file_path':'data/mimic_iii/test_dump/decay_data_20926.npz',
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
        # ml_trainer.run()

        model_reports.update(ml_trainer.get_reports())

        #DL Models
        model_type = ['LSTM','GRU', 'RIMDecay', 'RIM']
        for model in model_type:
            if model.startswith('RIM'):
                cell_type = ['LSTM', 'GRU']
            elif model == 'LSTM':
                cell_type = ['LSTM']
            else:
                cell_type = ['GRU']

            for cell in cell_type:
                args['rnn_cell'] = cell
                args['model_type'] = model
                if args['model_type'] == 'RIMDecay':
                    dl_trainer = TrainModels(args,decay_data_object)
                else:
                    dl_trainer = TrainModels(args,data_object)

                train_res =  dl_trainer.train()
                for model_1, res_sets in train_res.items():
                    y_truth, y_pred, y_score = res_sets
                    report = MIMICReport(model_1, y_truth, y_pred, y_score, './figures')
                    model_reports.update({model_1:report})
    else:
        #ML Test
        ml_trainer = MimicMlTrain(data_object, './mimic/models', OUTPUT)
        model_reports.update(ml_trainer.test())

        #DL Test
        # model_type = ['RIM_GRU', 'RIMDecay_GRU','RIM_LSTM', 'RIMDecay_LSTM','LSTM_GRU', 'GRU_LSTM']
        model_type = ['RIMDecay_GRU','RIMDecay_LSTM']
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
    
    # plot CMs and AUROC, AUPRC and...
    #Save the results to excel or latex 
    # plot the training curve for DL models 
    with open(f'./mimic/model_reports/reports.pickle', 'wb') as f:
        pickle.dump(model_reports, f)
    
    df_results =  pd.DataFrame(results)
    df_results = df_results.T
    print(df_results.to_markdown())
    plot_confusion_matrixes(model_reports)
    

def plot_prauc():
    with open(f'./mimic/model_reports/reports.pickle', 'rb') as f:
        reports = pickle.load(f)
    
    flag = True
    for model_name, report in reports.items():
        
        precision, recall, avg, name = report.get_prc_curve()
       
        plt.plot(precision, recall, label=model_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

def plot_roc():
    with open(f'./mimic/model_reports/reports.pickle', 'rb') as f:
        reports = pickle.load(f)
    
    flag = True
    for model_name, report in reports.items():
        if flag:
           ns_fpr, ns_tpr = no_skil(report.y_true) 
           plt.plot(ns_fpr, ns_tpr, marker='.', label='no_skill')
           flag=False
        fpr, tpr, _ = report.get_roc_curve()
        plt.plot(fpr, tpr, marker='.', label=model_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
def no_skil(y_true):
    
    ns_probs = [0 for _ in range(len(y_true))]
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
    return ns_fpr, ns_tpr



def plo_training_stats():
    from os import walk
    epocs = range(1,3)
    f = []
    losstats = []
    all_logs = {}
    for (_, _, filenames) in walk('./mimic/logs'):
        f.extend(filenames)
    
    for file in f:
        tmp = file.split(".")[0].split("_")[-2:]
        key = '_'.join(tmp)
        
        if key not in all_logs.keys():
            all_logs[key] = [file]
        else:
            all_logs[key].append(file)

    for key in all_logs.keys():
        plot_stats(key.split("_"),all_logs[key])

def plot_stats(key,stats):
    plt.figure()
    for file in stats:    
        with open(f'./mimic/logs/{file}', 'rb') as pickle_file:
            content = pickle.load(pickle_file)
            y = [i[-1] for i in content]
            epocs = range(1,len(content)+1)
            file_name = file.split('_')
            plt.plot(epocs,y, label = f'{file_name[0]}-{file_name[1]}' )
        
            plt.xlabel('Epocs')
            plt.ylabel(key[1])
            plt.title(f'{key[1]} stats for {key[0]}')  
    plt.legend()
    plt.show()
        

def plot_confusion_matrixes(reports):
       
    fig = plt.figure()
    gs = fig.add_gridspec(3, 3, hspace=10, wspace=10)
    axs = gs.subplots(sharex='col', sharey='row')
    fig.suptitle('Confusion matrix !')
    i = 0
    count = 0
    for model_name , report in reports.items():
        if count > 2:
            i +=1
            count =0
        axs[i,count % 3] = report.plot_cm().plot()
        count +=1
    fig.tight_layout()
    


mimic_main("train")
plo_training_stats()
# plot_roc()
# plot_prauc()
