import pickle

import numpy as np
import scipy.stats as ss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.model.mimic_ml_models import MimicMlTrain
from src.model.mimic_randomforrest import MIMICRF, MIMICLR
from src.utils.data_prep import MortalityDataPrep
from src.utils.mimic_evaluation import MIMICReport
from src.utils.mimic_iii_data import MIMICIIIData


'''
!!!! Important !!!!!!!
Input arguments like model_type and cell type are crucial.
'''
args = {
    'batch_size': 64,
    'static_features': 17,  # automatically picked from data
    'need_data_preprocessing': False,
    'raw_data_file_path': 'data/all_hourly_data_30000.pkl',
    'processed_data_path': 'data',
    'input_file_path': 'data/mimic_iii/test_dump/x_y_statics_20926.npz'
}
output = 'output/plots'


# Data preprocessing
if (args['need_data_preprocessing']):
    prep_data = MortalityDataPrep(args['raw_data_file_path'])
    _, _, _, args['input_file_path'] = prep_data.preprocess(True, args['processed_data_path'])
    del _

# data loader
data = MIMICIIIData(args['batch_size'], 24, args['input_file_path'], False)
args['input_size'] = data.input_size
args['static_features'] = data.static_features_size

train_x, dev_x, test_x, train_y, dev_y, test_y = data.get_ml_dataset()
# models + hyper parameters



ml_trainer = MimicMlTrain(data,'./mimic/models',output,3)
ml_trainer.run()

reports = ml_trainer.get_reports()
# reports = ml_trainer.test()
for model_name, model_report in reports.items():
    print(model_report.get_all_metrics())


# results = {}
# best_models = {}
# c_reports= {}
# for model_name, model, hyperparams_list in [  MIMICRF().get_model(),
#     MIMICLR().get_model() ]:
#     print("Running model %s on mortality prediction " % model_name)
#     best_model, best_hyperparameter, report = run_basic(model, hyperparams_list, train_x,dev_x, test_x,train_y,dev_y,test_y   )
#     results[model_name] = report
#     best_models[model_name] = [best_model,best_hyperparameter]
#
# for cn, rep in results.items():
#     print("Report for {}:" )
#     print(rep.get_all_metrics())
#     print(rep.get_sk_report())
#     print(rep.get_confusion_matrix())
#
# #write the best models into the disk
# with open(output+"classifires_results", mode='wb') as f:
#     pickle.dump(results, f)





# print(df_table.to_latex(float_format="%.3f"))

# table_report = TablePlot(df_table, output_dir=output)
# table_report.save_to_excel()
# table_report.save_to_latex()
# table_report.draw_table()

# print(df_table.to_markdown())

