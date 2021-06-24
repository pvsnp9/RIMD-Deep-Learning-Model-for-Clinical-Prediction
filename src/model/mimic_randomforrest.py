import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.ml_hyper_parameters import DictDist, Choice
import scipy.stats as ss

SEED =1



class MIMICRF():

    def __init__(self, N=5):
        self.N =N
        self.model_name = "RandomForest"
        RF_dist = DictDist({
            'n_estimators': ss.randint(50, 500),
            'max_depth': ss.randint(2, 10),
            'min_samples_split': ss.randint(2, 75),
            'min_samples_leaf': ss.randint(1, 50),
        })
        np.random.seed(SEED)
        self.RF_hyperparams_list = RF_dist.rvs(self.N)

    def get_model(self):
        return (self.model_name, RandomForestClassifier, self.RF_hyperparams_list)

class MIMICLR():

    def __init__(self, N=5):
        self.N = N
        self.model_name = "LogisticRegression"
        LR_dist = DictDist({
            'C': Choice(np.geomspace(1e-3, 1e3, 10000)),
            'penalty': Choice(['l1', 'l2']),
            'solver': Choice(['liblinear', 'lbfgs']),
            'max_iter': Choice([100, 500])
        })
        np.random.seed(SEED)
        self.LR_hyperparams_list = LR_dist.rvs(self.N)
        for i in range(self.N):
            if self.LR_hyperparams_list[i]['solver'] == 'lbfgs': self.LR_hyperparams_list[i]['penalty'] = 'l2'

    def get_model(self):
        return (self.model_name, LogisticRegression, self.LR_hyperparams_list)

