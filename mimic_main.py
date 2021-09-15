"""
!!!! Important !!!!!!!
Input arguments like model_type and cell type are crucial.
"""
from datetime import datetime

import pandas as pd
import pickle

import torch

from mimic_iii_train import TrainModels
from src.utils.mimic_args import get_args
from src.utils.mimic_args import common_args
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

# 1048
# 1024
np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)

N_HYPER_PARAM_SET = 10

SAVE_DIR = 'mimic/'


def mimic_main(run_type, run_description, out_dir):
    # Data preprocessing
    if (common_args['need_data_preprocessing']):
        prep_data = MortalityDataPrep(common_args['raw_data_file_path'])
        if common_args['mask']:
            common_args['input_file_path'] = prep_data.preprocess_decay(True, common_args['processed_data_path'])
        else:
            _, _, _, common_args['input_file_path'] = prep_data.preprocess(True, common_args['processed_data_path'])
            del _

    # https://github.com/manashty/lifemodel/blob/master/LifeModelForecasting/FallDetection.ipynb
    #
    ## Craet a directory for saving the results
    if run_type =="train":
        out_dir = MimicSave.get_instance().create_get_output_dir(SAVE_DIR)
    else:
        out_dir  = MimicSave.get_instance().create_get_output_dir(out_dir, is_test=True)
    # Save the args used in this experiment
    with open(f'{out_dir}/_experiment_args.txt', 'w') as f:
        json.dump(common_args, f)

    # config logging
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
    logging.info(f"out put directoru is { out_dir}")

    startTime = datetime.now()
    logging.info('Start time: ' + str(startTime))

    # load datasets
    decay_data_object = MIMICDecayData(common_args['batch_size'], 24, common_args['decay_input_file_path'])
    data_object = MIMICIIIData(common_args['batch_size'], 24, common_args['input_file_path'], common_args['mask'])

    model_reports = {}

    if run_type == "train":
        # ML models first
        ml_trainer = MimicMlTrain(data_object, './mimic/models', out_dir, logging, N_HYPER_PARAM_SET)
        # ml_trainer.run()
        # model_reports.update(ml_trainer.get_reports())
        # beta = [  0.59, 0.62, 0.65, 0.7, 0.73, 0.78,.79, 0.8, .83, 0.85, 0.9, 0.93, 0.95, 0.98]
        beta = [0.95]
        # DL Models
        for b in beta:

            model_type =['LSTM', 'GRU', 'RIM','RIMDecay','GRUD', 'RIMDCB']#['RIMDCB']  #
            for model in model_type:
                args = get_args(model)
                data_object.set_batch_size(args['batch_size'])
                decay_data_object.set_batch_size(args['batch_size'])
                args['cb_beta'] = b
                if model.startswith('RIM'):
                    cell_type = ['LSTM','GRU']

                elif model == 'LSTM':
                    cell_type = ['LSTM']
                elif model == 'GRUD':
                    cell_type = ['GRU']
                else:
                    cell_type = ['GRU']

                if model.endswith('CB'):
                    args['is_cbloss'] = True
                    model = 'RIMDecay'
                for cell in cell_type:
                    # TODO calculate execution time and log it
                    if args['is_cbloss']:
                        args['cb_beta'] = 0.93 if cell == 'GRU' else 0.95
                    args['rnn_cell'] = cell
                    args['model_type'] = model
                    if args['model_type'] == 'RIMDecay' or args['model_type'] == 'GRUD':
                        dl_trainer = TrainModels(args, decay_data_object, logging)
                    else:
                        dl_trainer = TrainModels(args, data_object, logging)

                    train_res = dl_trainer.train()
                    for model_1, res_sets in train_res.items():
                        y_truth, y_pred, y_score = res_sets
                        report = MIMICReport(model_1, y_truth, y_pred, y_score, './figures', args['is_cbloss'])
                        model_reports.update({model_1: report})


    else:
        # ML Test
        # ml_trainer = MimicMlTrain(data_object, './test_models/model', out_dir, logging)
        # model_reports.update(ml_trainer.test())

        # DL Test

        model_type = ['RIMDecay_GRU', 'RIMDecay_LSTM', "LSTM_LSTM", "GRU_GRU", 'RIM_GRU',
                      'RIM_LSTM','GRUD_GRU','RIMDecay_LSTM_cbloss', 'RIMDecay_GRU_cbloss']



        for model in model_type:
            if model.startswith('RIMDecay') or model.startswith('GRUD'):
                _,_, test_data = decay_data_object.data_loader()
            else:
                _,_,test_data = data_object.data_loader()
            trainer = TrainModels(logger=logging)
            model_path = f"./test_models/model/{model}_model.pt"
            cbloss = False
            if model.endswith('loss'):
                cbloss =True
            if not cbloss:
                y_truth, y_pred, y_score = trainer.test(model_path, test_data)
            else:
                y_truth, y_pred, y_score = trainer.test_cb_loss(model_path, test_data)
            report = MIMICReport(model, y_truth, y_pred, y_score, './figures', cbloss)
            model_reports.update({model: report})

    results = {}
    cms = {}
    for model, report in model_reports.items():
        results.update(report.get_all_metrics())
        cms.update({model: report.get_confusion_matrix()})

    # plot CMs and AUROC, AUPRC and...
    # Save the results to excel or latex
    # plot the training curve for DL models 

    df_results = pd.DataFrame(results)
    df_results = df_results.T
    # save to excel file
    print(df_results.to_markdown())
    plot_confusion_matrixes(out_dir, model_reports)

    """
    Saving the results
    """
    df_results.to_excel(f'{out_dir}/report.xlsx')
    df_results.to_latex(f'{out_dir}/report.tex')
    with open(f'{out_dir}/reports.pickle', 'wb') as f:
        pickle.dump(model_reports, f)


def plot_prauc(experiment_address=None):
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
            flag = False
        fpr, tpr, _ = report.get_roc_curve()
        plt.plot(fpr, tpr, label=model_name)
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
    epocs = range(1, 3)
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
        plot_stats(key.split("_"), all_logs[key])


def plot_stats(key, stats):
    plt.figure()
    for file in stats:
        with open(f'{MimicSave.get_instance().get_log_directory()}/{file}', 'rb') as pickle_file:
            content = pickle.load(pickle_file)
            y = [i[-1] for i in content]
            epocs = range(1, len(content) + 1)
            file_name = file.split('_')
            plt.plot(epocs, y, label=f'{file_name[0]}-{file_name[1]}')

            plt.xlabel('Epocs')
            plt.ylabel(key[1])
            plt.title(f'{key[1]} stats for {key[0]}')
    plt.legend()
    # plt.show()
    plt.savefig(f'{MimicSave.get_instance().get_directory()}/{file.split("_")[0]}_{key[0]}_{key[1]}', dpi=300)
    plt.plot()
    plt.show()


def plot_confusion_matrixes(out_dir, reports):
    fig = plt.figure()
    gs = fig.add_gridspec(3, 3, hspace=10, wspace=10)
    axs = gs.subplots(sharex='col', sharey='row')
    fig.suptitle('Confusion matrix !')
    i = 0
    count = 0
    for model_name, report in reports.items():
        if count > 2:
            i += 1
            count = 0
        axs[i, count % 3] = report.plot_cm(f'{out_dir}/CM_{model_name}.png').plot()
        count += 1
    fig.tight_layout()


MimicSave.get_instance()

description = "Experiment # 2:   tuned models "
out_dir = './test_models/results/'
mimic_main("train", description, out_dir)

plo_training_stats()
plot_roc(out_dir)
plot_prauc(out_dir)
