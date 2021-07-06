import datetime
import pickle

import numpy as np
import scipy.stats as ss
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.model.mimic_randomforrest import MIMICLR, MIMICRF
from src.utils.data_prep import MortalityDataPrep
from src.utils.mimic_evaluation import MIMICReport
from src.utils.mimic_iii_data import MIMICIIIData


class MimicMlTrain():


    def __init__(self, data_object, save_dir, output_dir,logger, N_hyper_set=5):
        self.data_object = data_object
        self.best_hyper_parameters= {}
        self.reports = {}
        self.best_models = {}
        self.trained_model = {}
        self.save_dir = save_dir
        self.output_dir = output_dir
        self.logger = logger

        #Define the classification models here
        self.ml_models = [MIMICRF(N_hyper_set).get_model(), MIMICLR(N_hyper_set).get_model()]

    def run(self):
        self.logger.info("\t##########################################################################")
        self.logger.info("\t############# Trining ML Models  {} #################### ".format(datetime.datetime.now()) )
        self.logger.info("\t##########################################################################")
        self.logger.info('')

        train_x, dev_x, test_x, train_y, dev_y, test_y = self.data_object.get_ml_dataset()
        for model_name, model, hyperparams_list in self.ml_models :

            self.logger.info("Running model %s on mortality prediction " % model_name)
            self.logger.info("")
            report = self.train(model_name, model, hyperparams_list, train_x, dev_x, test_x,
                                train_y, dev_y, test_y)

            self.reports[model_name] = report

    def get_reports(self):
        return self.reports

    def train(self, model_name, model, hyperparams_list, X_flat_train, X_flat_dev, X_flat_test, Ys_train, Ys_dev, Ys_test):

        best_s, best_hyperparams = -np.Inf, None
        for i, hyperparams in enumerate(hyperparams_list):

            self.logger.info("On sample %d / %d (hyperparams = %s)" % (i + 1, len(hyperparams_list), repr((hyperparams))))
            M = model(**hyperparams)
            M.fit(X_flat_train, Ys_train)
            s = roc_auc_score(Ys_dev, M.predict_proba(X_flat_dev)[:, 1])
            if s > best_s:
                best_s, best_hyperparams = s, hyperparams
                self.logger.info("New Best Score: %.2f @ hyperparams = %s" % (100 * best_s, repr((best_hyperparams))))

        return self.train_best_model(model_name, model, best_hyperparams, X_flat_train, X_flat_dev, X_flat_test, Ys_train, Ys_dev,
                                     Ys_test)

    def train_best_model(self, model_name, model, best_hyperparams, X_flat_train, X_flat_dev, X_flat_test, Ys_train, Ys_dev, Ys_test):
        best_M = model(**best_hyperparams)
        best_M.fit(np.concatenate([X_flat_train, X_flat_dev]), np.concatenate([Ys_train, Ys_dev]))
        y_true = Ys_test
        # y_score = best_M.predict_proba(X_flat_test)[:, 1]
        # if hasattr(best_M, "decision_function"):
        #     y_score = best_M.decision_function(X_flat_test)
        # else:
        y_score = best_M.predict_proba(X_flat_test)[:, 1]

        y_pred = best_M.predict(X_flat_test)

        report = MIMICReport(model_name, y_true, y_pred, y_score, output_dir=self.output_dir)

        save_name = f'{self.save_dir}/{model_name}-model.joblib'
        dump(best_M, save_name)
        return report


    def test(self):
        test_x, test_y =  self.data_object.get_ml_test()
        for model_name, _,_ in self.ml_models:
            save_name = f'{self.save_dir}/{model_name}-model.joblib'
            model =  load(save_name)
            y_pred = model.predict(test_x)
            y_score = model.predict_proba(test_x)[:, 1]
            self.reports[model_name] = MIMICReport(model_name, test_y, y_pred, y_score, output_dir=self.output_dir)
        return self.reports









