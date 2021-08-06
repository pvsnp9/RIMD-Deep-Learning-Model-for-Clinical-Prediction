args = {'epochs': 50,
        'batch_size':56,
        'input_size': 104,
        'model_type': 'RIMDecay',
        'hidden_size': 104,
        'num_rims': 2,
        'lr': 0.001,#9.645142728701031e-05,
        'rnn_cell': 'GRU',
        'input_key_size': 64,
        'input_value_size': 32,
        'input_query_size': 64,
        'num_input_heads': 1,
        'input_dropout': 0.2694669454300502,
        'comm_key_size': 32,
        'comm_value_size': 104,
        'comm_query_size': 32,
        'num_comm_heads': 1,
        'comm_dropout': 0.12824679075614986,
        'active_rims': 2,
        'mask': True,
        'mask_size': 104,
        'delta_size': 104,
        'static_features': 16,
        'need_data_preprocessing': False,
        'raw_data_file_path': 'data/10_percent/all_hourly_data.pkl',
        'processed_data_path': 'data/50_percent/',
        'input_file_path': 'data/mimic_iii/test_dump/decay_x_y_statics_xmean_mask_delta_lastob_los_icu_23944.npz',
        'decay_input_file_path': 'data/mimic_iii/test_dump/decay_x_y_statics_xmean_mask_delta_lastob_los_icu_23944.npz',
        'max_no_improvement': 20,
        'improvement_threshold': 0.0001,
        'is_tuning': False,
        'balance': False
        , 'is_cbloss': False
}


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