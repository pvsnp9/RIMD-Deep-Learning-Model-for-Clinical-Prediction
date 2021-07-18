
import optuna
from optuna.trial import TrialState
import math

from mimic_iii_train import TrainModels
from src.mimic_args import args
import logging
#load datasets
from src.utils.mimic_iii_data import MIMICIIIData
from src.utils.mimic_iii_decay_data import MIMICDecayData

out_dir = 'mimic/logs'
# config logging
logging.basicConfig(filename=out_dir + '/' + 'tuning-log.txt', format='%(message)s', level=logging.DEBUG)
# Adding log to console as well
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logging.Formatter('*\t%(message)s'))
logging.getLogger().addHandler(consoleHandler)

def draw_args(trial):
    """
    Here we will randomly generate args values before each model training
    """
    args['batch_size'] = 48 #trial.suggest_int('batch_size',low=32,high=96,step=16)

    # args['hidden_size'] =  trial.suggest_int('hidden_size', low=30,high=100,step=10)
    # args['comm_value_size'] = args['hidden_size']


    args['num_rims'] =  trial.suggest_int('num_rims', low=2,high=6,step=1)
    args['active_rims'] = math.ceil(0.6 * args['num_rims'])


    args['lr'] =  trial.suggest_float('lr', low=0.00001 ,high= 0.001)

    args['input_key_size'] =  trial.suggest_categorical('input_key_size', [32,64,128])
    args['input_query_size'] = args['input_key_size']

    args['comm_key_size'] = trial.suggest_categorical('comm_key_size', [32, 64, 128])
    args['comm_query_size'] = args['comm_key_size']

    args['input_value_size'] =  trial.suggest_categorical('input_value_size', [32,64,128])

    args['num_input_heads'] =  trial.suggest_categorical('num_input_heads', [1,2])
    args['num_comm_heads'] = args['num_input_heads']

    args['input_dropout'] =  trial.suggest_float('input_dropout', low=0.2 ,high= 0.4)
    args['comm_dropout'] =  trial.suggest_float('comm_dropout', low=0.1 ,high= 0.15)
    args['is_tuning'] = True
    args['balance'] = False


def tuning_model(trial):
    """
    iterate over the models
    """
    draw_args(trial)
    print(args)
    decay_data_object = MIMICDecayData(args['batch_size'], 24, args['decay_input_file_path'], balance=args['balance'])
    # data_object = MIMICIIIData(args['batch_size'], 24, args['input_file_path'], args['mask'])
    dl_trainer = TrainModels(args, decay_data_object, logging)


    train_res = dl_trainer.tune_train()

        # train_res =0
        # logging.info("Something went wrong with trials number {}".format(trial))
    return train_res



def save_best_parameters():
    pass

if __name__ == '__main__':
    args['rnn_cell'] = 'GRU'
    args['model_type'] = 'RIMDecay'
    study = optuna.create_study(direction="maximize")
    study.optimize(tuning_model, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
