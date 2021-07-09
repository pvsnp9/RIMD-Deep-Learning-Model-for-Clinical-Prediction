args = {
    'epochs':35,
    'batch_size': 48,
    'input_size': 1, #automatically picked from data
    'model_type': 'RNN', # type of model  RIM, LSTM, GRU
    'hidden_size': 50,
    'num_rims': 8,
    'lr': 0.001756254208325344,
    'rnn_cell': 'GRU_D', # type of cell LSTM, or GRU
    'input_key_size': 128,
    'input_value_size': 128,
    'input_query_size': 128,
    'num_input_heads': 3,
    'input_dropout': 0.1638808451502491,
    'comm_key_size': 128,
    'comm_value_size': 50,
    'comm_query_size': 128,
    'num_comm_heads': 3,
    'comm_dropout': 0.15,
    'active_rims': 5,
    'mask': True,
    'mask_size': 104,
    'delta_size': 104,
    'static_features':17, #automatically picked from data

    'need_data_preprocessing': False,
    'raw_data_file_path' :'data/10_percent/all_hourly_data.pkl',
    'processed_data_path':'data/50_percent/',
    'input_file_path':'data/mimic_iii/test_dump/x_y_statics_20926.npz',
    'decay_input_file_path':'data/mimic_iii/test_dump/decay_data_20926.npz',
    'max_no_improvement' : 10,
    'improvement_threshold' : 0.0001,
    'is_tuning' :  False,
    'balance' : False
}
