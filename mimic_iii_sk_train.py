import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, classification_report, \
    precision_score
from tqdm import tqdm
import pickle
import numpy as np
from src.model.mimic_model import MIMICModel
from src.model.mimic_lstm_model import MIMICLSTMModel
from src.model.mimic_gru_model import MIMICGRUModel
from src.utils.mimic_class_report import ClassificationReport, TablePlot
from src.utils.mimic_iii_data import MIMICIIIData
from src.utils.data_prep import MortalityDataPrep

import copy, math, os, time, pandas as pd, scipy.stats as ss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    'input_file_path': 'data/x_y_statics_20926.npz'
}
output = 'output/plots'


SEED = 1

np.random.seed(SEED)


class DictDist():
    def __init__(self, dict_of_rvs):
        self.dict_of_rvs = dict_of_rvs

    def rvs(self, n):
        a = {k: v.rvs(n) for k, v in self.dict_of_rvs.items()}
        out = []
        for i in range(n): out.append({k: vs[i] for k, vs in a.items()})
        return out


class Choice():
    def __init__(self, options):
        self.options = options

    def rvs(self, n):
        return [self.options[i] for i in ss.randint(0, len(self.options)).rvs(n)]


def run_basic(model, hyperparams_list, X_flat_train, X_flat_dev, X_flat_test, Ys_train, Ys_dev, Ys_test):
    best_s, best_hyperparams = -np.Inf, None
    for i, hyperparams in enumerate(hyperparams_list):
        print("On sample %d / %d (hyperparams = %s)" % (i + 1, len(hyperparams_list), repr((hyperparams))))
        M = model(**hyperparams)
        M.fit(X_flat_train, Ys_train)
        s = roc_auc_score(Ys_dev, M.predict_proba(X_flat_dev)[:, 1])
        if s > best_s:
            best_s, best_hyperparams = s, hyperparams
            print("New Best Score: %.2f @ hyperparams = %s" % (100 * best_s, repr((best_hyperparams))))

    return run_only_final(model, best_hyperparams, X_flat_train, X_flat_dev, X_flat_test, Ys_train, Ys_dev, Ys_test )


def run_only_final(model, best_hyperparams, X_flat_train, X_flat_dev, X_flat_test, Ys_train, Ys_dev, Ys_test):
    best_M = model(**best_hyperparams)
    best_M.fit(np.concatenate    ([X_flat_train, X_flat_dev]), np.concatenate([Ys_train, Ys_dev]))
    y_true = Ys_test
    # y_score = best_M.predict_proba(X_flat_test)[:, 1]
    if hasattr(best_M, "decision_function"):
        y_score = best_M.decision_function(X_flat_test)
    else:
        y_score = best_M.predict_proba(X_flat_test)[:, 1]
    y_pred = best_M.predict(X_flat_test)

    report = ClassificationReport(best_M, y_true, y_pred,y_score, output_dir=output)
    res_ = report.get_all_metrics()
    report.save_plots(X_flat_test, output)
    report.plot_calibration_curve(fig_index=1, X_train=np.concatenate([X_flat_train, X_flat_dev]), X_test=X_flat_test, y_train=np.concatenate([Ys_train, Ys_dev]), y_test=y_true)

    return best_M, best_hyperparams, res_, classification_report(y_true,y_pred)


# Data preprocessing
if (args['need_data_preprocessing']):
    prep_data = MortalityDataPrep(args['raw_data_file_path'])
    _, _, _, args['input_file_path'] = prep_data.preprocess(True, args['processed_data_path'])
    del _

# data loader
data = MIMICIIIData(args['batch_size'], 24, args['input_file_path'], False)
args['input_size'] = data.input_size()
args['static_features'] = data.static_features_size()

train_x, dev_x, test_x, train_y, dev_y, test_y = data.get_sk_dataset()
# models + hyper parameters

N = 15

LR_dist = DictDist({
    'C': Choice(np.geomspace(1e-3, 1e3, 10000)),
    'penalty': Choice(['l1', 'l2']),
    'solver': Choice(['liblinear', 'lbfgs']),
    'max_iter': Choice([100, 500])
})
np.random.seed(SEED)
LR_hyperparams_list = LR_dist.rvs(N)
for i in range(N):
    if LR_hyperparams_list[i]['solver'] == 'lbfgs': LR_hyperparams_list[i]['penalty'] = 'l2'

RF_dist = DictDist({
    'n_estimators': ss.randint(50, 500),
    'max_depth': ss.randint(2, 10),
    'min_samples_split': ss.randint(2, 75),
    'min_samples_leaf': ss.randint(1, 50),
})
np.random.seed(SEED)
RF_hyperparams_list = RF_dist.rvs(N)

results = {}
best_models = {}
c_reports= {}
for model_name, model, hyperparams_list in [ ('RF', RandomForestClassifier, RF_hyperparams_list),
    ('LR', LogisticRegression, LR_hyperparams_list) ]:
    print("Running model %s on mortality prediction " % model_name)
    best_model, best_hyperparameter, result, rep= run_basic(model, hyperparams_list, train_x,dev_x, test_x,train_y,dev_y,test_y   )
    results.update(result)
    best_models[model_name] = [best_model,best_hyperparameter]
    c_reports[model_name] = rep


#write the best models into the disk
with open(output+"classifires_results", mode='wb') as f:
    pickle.dump(results, f)


df_table = pd.DataFrame(results)
df_table = df_table.T

s = df_table.style
s.highlight_max(axis=1)
df_table.style.apply(s)


# print(df_table.to_latex(float_format="%.3f"))

table_report = TablePlot(df_table, output_dir=output)
table_report.save_to_excel()
table_report.save_to_latex()
table_report.draw_table()

print(df_table.to_markdown())

for cn, rep in c_reports.items():
    print("{} : \n {}".format(cn,rep))