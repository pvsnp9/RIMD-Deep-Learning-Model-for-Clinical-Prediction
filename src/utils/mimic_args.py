import copy

args = {'epochs': 100, 'batch_size': 80, 'input_size': 104, 'model_type': 'RIMDecay', 'hidden_size': 104, 'num_rims': 5,
        'lr': 0.0003035962616430898, 'rnn_cell': 'LSTM', 'input_key_size': 32, 'input_value_size': 32,
        'input_query_size': 32, 'num_input_heads': 1, 'input_dropout': 0.23725840765323997, 'comm_key_size': 128,
        'comm_value_size': 104, 'comm_query_size': 128, 'num_comm_heads': 1, 'comm_dropout': 0.10870214577582729,
        'active_rims': 3, 'mask': True, 'mask_size': 104, 'delta_size': 104, 'static_features': 16,
        'need_data_preprocessing': False, 'raw_data_file_path': 'data/10_percent/all_hourly_data.pkl',
        'processed_data_path': 'data/50_percent/', 'input_file_path': 'data/mimic_iii/test_dump/decay_data_20926.npz',
        'decay_input_file_path': 'data/mimic_iii/test_dump/decay_data_20926.npz', 'max_no_improvement': 25,
        'improvement_threshold': 0.0001, 'is_tuning': True, 'balance': False, 'is_cbloss': False, 'cb_beta': 0.1}

# {'epochs': 100, 'batch_size': 32, 'input_size': 104, 'model_type': 'RIMDecay', 'hidden_size': 104, 'num_rims': 5, 'lr': 0.00020303449512190297, 'rnn_cell': 'LSTM', 'input_key_size': 128, 'input_value_size': 32, 'input_query_size': 128, 'num_input_heads': 2, 'input_dropout': 0.3737847512381478, 'comm_key_size': 64, 'comm_value_size': 104, 'comm_query_size': 64, 'num_comm_heads': 2, 'comm_dropout': 0.10913199819628257, 'active_rims': 3, 'mask': True, 'mask_size': 104, 'delta_size': 104, 'static_features': 16, 'need_data_preprocessing': False, 'raw_data_file_path': 'data/10_percent/all_hourly_data.pkl', 'processed_data_path': 'data/50_percent/', 'input_file_path': 'data/mimic_iii/test_dump/decay_data_20926.npz', 'decay_input_file_path': 'data/mimic_iii/test_dump/decay_data_20926.npz', 'max_no_improvement': 25, 'improvement_threshold': 0.0001, 'is_tuning': True, 'balance': False, 'is_cbloss': False, 'cb_beta': 0.59}

common_args = {
        'epochs': 100,
        "batch_size": 80,
        'static_features': 16,
        'need_data_preprocessing': False,
        'raw_data_file_path': 'data/mimic_iii/test_dump/decay_data_20926.npz',
        'processed_data_path': 'data/50_percent/',
        'input_file_path': 'data/mimic_iii/test_dump/decay_data_20926.npz',
        'decay_input_file_path': 'data/mimic_iii/test_dump/decay_data_20926.npz',
        'max_no_improvement': 35,
        'improvement_threshold': 0.0001,
        'is_tuning': False,
        'balance': False,
        'is_cbloss': False,
        'cb_beta': 0.59,
        'mask': True,
        'mask_size': 104,
        'delta_size': 104,
}
tuned_args = {
    'LSTM': {
        "batch_size": 80,
        "hidden_size": 40,
        'input_size': 104,
        "lr": 0.00018042495392167962,
        'input_dropout': 0.20816385459954448
    },
    'GRU': {
        "batch_size": 64,
        "hidden_size": 40,
        'input_size': 104,
        "lr": 6.549427303704413e-05,
        'input_dropout': 0.20816385459954448

    },
    "RIM":
        {
            'batch_size': 48,
            'input_size': 104,
            'hidden_size': 100,
            'comm_value_size': 100,
            'num_rims': 5,
            'active_rims': 3,
            'lr': 0.000805983139352919,
            'input_key_size': 128,
            'input_query_size': 128,
            'comm_key_size': 128,
            'comm_query_size': 128,
            'input_value_size': 32,
            'num_input_heads':1,
            'num_comm_heads': 1,
            'input_dropout': 0.2554497828446886,
            'comm_dropout': 0.1197668921246133
        },
    "RIMDecay": {
        'epochs': 100,
        'batch_size': 80,
        'input_size': 104,
        'model_type': 'RIMDecay',
        'hidden_size': 104,
        'num_rims': 5,
        'lr': 0.00020303449512190297,
        'rnn_cell': 'LSTM',
        'input_key_size': 128,
        'input_value_size': 32,
        'input_query_size': 128,
        'num_input_heads': 2,
        'input_dropout': 0.3737847512381478,
        'comm_key_size': 64,
        'comm_value_size': 104,
        'comm_query_size': 64,
        'num_comm_heads': 2,
        'comm_dropout': 0.10913199819628257,
        'active_rims': 3,
        'mask': True,
        'mask_size': 104,
        'delta_size': 104,
    },
    "GRUD": {
        'batch_size': 32,
        'hidden_size': 40,
        'lr': 3.0227696576190026e-05,
        'input_dropout': 0.20816385459954448},
    "RIMDCB": {
        'epochs': 100,
        'batch_size': 48,
        'input_size': 104,
        'model_type': 'RIMDecay',
        'hidden_size': 104,
        'num_rims': 3,
        'lr': 0.00020303449512190297,
        'rnn_cell': 'LSTM',
        'input_key_size': 128,
        'input_value_size': 32,
        'input_query_size': 128,
        'num_input_heads': 2,
        'input_dropout': 0.3737847512381478,
        'comm_key_size': 64,
        'comm_value_size': 104,
        'comm_query_size': 64,
        'num_comm_heads': 2,
        'comm_dropout': 0.10913199819628257,
        'active_rims': 2,
        'mask': True,
        'mask_size': 104,
        'delta_size': 104,
    }
}


def get_args(model_name):
    args = copy.deepcopy(common_args)

    args.update(tuned_args[model_name])
    return  args

    # for 60 percent PRAUC {'epochs': 100,
    # 'batch_size': 96,
    # 'input_size': 104,
    # 'model_type': 'RIMDecay',
    # 'hidden_size': 60,
    # 'num_rims': 2,
    # 'lr': 0.0007614240581092288,
    # 'rnn_cell': 'LSTM',
    # 'input_key_size': 32,
    # 'input_value_size': 64,
    # 'input_query_size': 32,
    # 'num_input_heads': 1,
    # 'input_dropout': 0.31539390117123534,
    # 'comm_key_size': 32,
    # 'comm_value_size': 104,
    # 'comm_query_size': 32,
    # 'num_comm_heads': 1,
    # 'comm_dropout': 0.12571791902680474,
    # 'active_rims': 2, 'mask': True,
    # 'mask_size': 104, 'delta_size': 104,
    # 'static_features': 16, 'need_data_preprocessing': False,
    # 'raw_data_file_path': 'data/10_percent/all_hourly_data.pkl',
    # 'processed_data_path': 'data/50_percent/',
    # 'input_file_path': 'data/mimic_iii/test_dump/decay_data_20926.npz',
    # 'decay_input_file_path': 'data/mimic_iii/test_dump/decay_data_20926.npz',
    # 'max_no_improvement': 30, 'improvement_threshold': 0.0001,
    # 'is_tuning': True,
    # 'balance': False,
    # 'is_cbloss': True,
    # 'cb_beta': 0.95}
# {'epochs': 100,
#         'batch_size':80,
#         'input_size': 104,
#         'model_type': 'RIMDecay',
#         'hidden_size': 30,
#         'num_rims': 6,
#         'lr': 1.2842105011782403e-05,
#         'rnn_cell': 'GRU',
#         'input_key_size': 32,
#         'input_value_size': 32,
#         'input_query_size': 32,
#         'num_input_heads': 2,
#         'input_dropout': 0.2264497828446886,
#         'comm_key_size': 32,
#         'comm_value_size': 30,
#         'comm_query_size': 32,
#         'num_comm_heads': 2,
#         'comm_dropout': 0.1027998921246133,
#         'active_rims': 4,
#         'mask': True,
#         'mask_size': 104,
#         'delta_size': 104,
#         'static_features': 16,
#         'need_data_preprocessing': False,
#         'raw_data_file_path': 'data/10_percent/all_hourly_data.pkl',
#         'processed_data_path': 'data/50_percent/',
#         'input_file_path': 'data/mimic_iii/test_dump/decay_data_20926.npz',
#         'decay_input_file_path': 'data/mimic_iii/test_dump/decay_data_20926.npz',
#         'max_no_improvement': 25,
#         'improvement_threshold': 0.0001,
#         'is_tuning': False,
#         'balance': False
#         , 'is_cbloss': False,
#         'cb_beta' :0.59
#
#         }


#
# {
#     'epochs':35,
#     'batch_size': 48,
#     'input_size': 1, #automatically picked from data
#     'model_type': 'RNN', # type of model  RIM, LSTM, GRU
#     'hidden_size': 50,
#     'num_rims': 8,
#     'lr': 0.001756254208325344,
#     'rnn_cell': 'GRU_D', # type of cell LSTM, or GRU
#     'input_key_size': 128,
#     'input_value_size': 128,
#     'input_query_size': 128,
#     'num_input_heads': 3,
#     'input_dropout': 0.1638808451502491,
#     'comm_key_size': 128,
#     'comm_value_size': 50,
#     'comm_query_size': 128,
#     'num_comm_heads': 3,
#     'comm_dropout': 0.15,
#     'active_rims': 5,
#     'mask': True,
#     'mask_size': 104,
#     'delta_size': 104,
#     'static_features':17, #automatically picked from data
#
#     'need_data_preprocessing': False,
#     'raw_data_file_path' :'data/10_percent/all_hourly_data.pkl',
#     'processed_data_path':'data/50_percent/',
#     'input_file_path':'data/mimic_iii/test_dump/x_y_statics_20926.npz',
#     'decay_input_file_path':'data/mimic_iii/test_dump/decay_data_20926.npz',
#     'max_no_improvement' : 10,
#     'improvement_threshold' : 0.0001,
#     'is_tuning' :  False,
#     'balance' : False
# }
