"""
!!!! Important !!!!!!!
Input arguments like model_type and cell type are crucial.
"""
from datetime import datetime

import pandas as pd
import pickle

import torch

from mimic_iii_train import TrainModels
from src.utils.mimic_args import args
from src.model.mimic_ml_models import MimicMlTrain
from src.utils.data_prep import MortalityDataPrep
from src.utils.mimic_evaluation import MIMICReport
from src.utils.mimic_iii_data import MIMICIIIData
from src.utils.mimic_iii_decay_data import MIMICDecayData
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import json
from src.utils.save_utils import MimicSave
import logging
import numpy as np

np.random.seed(1048)
torch.manual_seed(1048)
torch.cuda.manual_seed(1048)


N_HYPER_PARAM_SET = 10

SAVE_DIR = 'mimic/'


def mimic_main(run_type, run_description):

    # Data preprocessing
    if (args['need_data_preprocessing']):
        prep_data = MortalityDataPrep(args['raw_data_file_path'])
        if args['mask']:
            args['input_file_path'] = prep_data.preprocess_decay(True, args['processed_data_path'])
        else:
            _, _, _, args['input_file_path'] = prep_data.preprocess(True, args['processed_data_path'])
            del _

    # https://github.com/manashty/lifemodel/blob/master/LifeModelForecasting/FallDetection.ipynb
    #
    ## Craet a directory for saving the results

    out_dir = MimicSave.get_instance().create_get_output_dir(SAVE_DIR)

    # Save the args used in this experiment
    with open(f'{out_dir}/_experiment_args.txt','w') as f:
        json.dump(args,f)

    #config logging
    logging.basicConfig(filename=out_dir + '/' + 'log.txt', format='%(message)s', level=logging.DEBUG)
    # Adding log to console as well
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter('*\t%(message)s'))
    logging.getLogger().addHandler(consoleHandler)
    logging.getLogger('matplotlib.font_manager').disabled = True
    dateformat = "%Y/%m/%d  %H:%M:%S"
    logging.info("Run Description: " + run_description)
    logging.info("Log file created at " + datetime.now().strftime("%Y/%m/%d  %H:%M:%S"))
    logging.info("Directory: {0}".format(out_dir))

    startTime = datetime.now()
    logging.info('Start time: ' + str(startTime))



    #load datasets
    decay_data_object = MIMICDecayData(args['batch_size'], 24, args['decay_input_file_path'])
    data_object = MIMICIIIData(args['batch_size'], 24, args['input_file_path'], args['mask'])
    
    model_reports = {}

    if run_type == "train":
        #ML models first
        ml_trainer = MimicMlTrain(data_object, './mimic/models', out_dir,logging, N_HYPER_PARAM_SET)
        # ml_trainer.run()
        # model_reports.update(ml_trainer.get_reports())

        #DL Models
        model_type = ['LSTM', 'GRU', 'RIMDecay','RIM']
        for model in model_type:
            if model.startswith('RIM'):
                cell_type = [ 'GRU','LSTM']
            elif model == 'LSTM':
                cell_type = ['LSTM']
            elif args['model_type'] == 'GRUD':
                cell_type = ['GRU']
            else:
                cell_type = ['GRU']

            for cell in cell_type:
                #TODO calculate execution time and log it
                args['rnn_cell'] = cell
                args['model_type'] = model
                if args['model_type'] == 'RIMDecay' or args['model_type'] == 'GRUD':
                    dl_trainer = TrainModels(args,decay_data_object, logging)
                else:
                    dl_trainer = TrainModels(args,data_object, logging)

                train_res =  dl_trainer.train()
                for model_1, res_sets in train_res.items():
                    y_truth, y_pred, y_score = res_sets
                    report = MIMICReport(model_1, y_truth, y_pred, y_score, './figures',args['is_cbloss'])
                    model_reports.update({model_1:report})


    elif(run_type == 'train_with_cb_loss'):
        logging.info("Training initiated with custom loss function")
        ml_trainer = MimicMlTrain(data_object, './mimic/models', out_dir,logging, N_HYPER_PARAM_SET)
        model_type = [ 'RIMDecay'] #, 'RIM' ]
        for model in model_type:
            if model.startswith('RIM'):
                cell_type = [ 'GRU','LSTM']
            elif model == 'LSTM':
                cell_type = ['LSTM']
            else:
                cell_type = ['GRU']

            for cell in cell_type:
                #TODO calculate execution time and log it
                args['rnn_cell'] = cell
                args['model_type'] = model
                if args['model_type'] == 'RIMDecay':
                    dl_trainer = TrainModels(args,decay_data_object, logging)
                else:
                    dl_trainer = TrainModels(args,data_object, logging)

                train_res =  dl_trainer.train_cb_loss()
                for model_1, res_sets in train_res.items():
                    y_truth, y_pred, y_score = res_sets
                    report = MIMICReport(model_1, y_truth, y_pred, y_score, './figures')
                    model_reports.update({model_1:report})
    else:
        #ML Test
        ml_trainer = MimicMlTrain(data_object, './mimic/0727-10-55-06/model', out_dir,logging)
        model_reports.update(ml_trainer.test())

        #DL Test
        # model_type = ['RIM_GRU', 'RIMDecay_GRU','RIM_LSTM', 'RIMDecay_LSTM','LSTM_GRU', 'GRU_LSTM']
        model_type = ['RIMDecay_GRU','RIMDecay_LSTM', "LSTM", "GRU"]
        cell_type = ['LSTM', 'GRU']

        for model in model_type:
            if model.startswith('RIMDecay'):
                test_data = decay_data_object.get_test_data()
            else:
                test_data = data_object.get_test_data()
            trainer = TrainModels(logger= logging)
            model_path = f"{SAVE_DIR}/{model}_model.pt"
            y_truth, y_pred, y_score = trainer.test(model_path, test_data)
            report = MIMICReport(model, y_truth, y_pred, y_score, './figures', args['is_cbloss'])
            model_reports.update({model:report})

    results = {}
    cms = {}
    for model, report in model_reports.items():
        results.update(report.get_all_metrics())
        cms.update({model:report.get_confusion_matrix()})
    
    # plot CMs and AUROC, AUPRC and...
    #Save the results to excel or latex 
    # plot the training curve for DL models 

    df_results =  pd.DataFrame(results)
    df_results = df_results.T
    # save to excel file
    print(df_results.to_markdown())
    plot_confusion_matrixes(out_dir,model_reports)

    """
    Saving the results
    """
    df_results.to_excel(f'{out_dir}/report.xlsx')
    df_results.to_latex(f'{out_dir}/report.tex')
    with open(f'{out_dir}/reports.pickle', 'wb') as f:
        pickle.dump(model_reports, f)


def plot_prauc( experiment_address = None):

    if experiment_address != None:
        folder_address = experiment_address
    else:
        folder_address = MimicSave.get_instance().get_directory()
    with open(f'{folder_address}/reports.pickle', 'rb') as f:
        reports = pickle.load(f)

    flag = True
    for model_name, report in reports.items():
        
        precision, recall, avg, name = report.get_prc_curve()
       
        plt.plot(precision, recall, label=model_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'{folder_address}/PRAUC_plot.png')
    plt.show()

def plot_roc(experiment_address=None):

    if experiment_address != None:
        folder_address = experiment_address
    else:
        folder_address = MimicSave.get_instance().get_directory()
    with open(f'{folder_address}/reports.pickle', 'rb') as f:
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
    plt.savefig(f'{folder_address}/ROC_plot.png')
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
    for (_, _, filenames) in walk(MimicSave.get_instance().get_log_directory()):
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
        with open(f'{MimicSave.get_instance().get_log_directory()}/{file}', 'rb') as pickle_file:
            content = pickle.load(pickle_file)
            y = [i[-1] for i in content]
            epocs = range(1,len(content)+1)
            file_name = file.split('_')
            plt.plot(epocs,y, label = f'{file_name[0]}-{file_name[1]}' )
        
            plt.xlabel('Epocs')
            plt.ylabel(key[1])
            plt.title(f'{key[1]} stats for {key[0]}')  
    plt.legend()
    # plt.show()
    plt.savefig(f'{MimicSave.get_instance().get_directory()}/{file.split("_")[0]}_{key[0]}_{key[1]}', dpi = 300)
    plt.plot()
    plt.show()

def plot_confusion_matrixes(out_dir,reports):
       
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
        axs[i,count % 3] = report.plot_cm(f'{out_dir}/CM_{model_name}.png').plot()
        count +=1
    fig.tight_layout()


MimicSave.get_instance()

description = "Experiment # 1.2: RIMDecay tuned "
mimic_main("train",description)
plo_training_stats()
plot_roc()
plot_prauc()

