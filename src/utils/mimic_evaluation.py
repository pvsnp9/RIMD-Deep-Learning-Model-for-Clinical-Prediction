from matplotlib import pyplot as plt
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve


class MIMICReport():

    def __init__(self,classifire_name, y_true, y_pred, y_score, output_dir):
        self.y_pred = y_pred
        self.y_true = y_true
        self.y_score = y_score
        self.y_max = max(y_true)
        self.output_dir = output_dir
        self.classifire_name = classifire_name
    """
    Classification Report : SK-Learn version 
    """
    def get_sk_report(self):
        return classification_report(self.y_true,self.y_pred)

    """
    Classification report for positive class only, as we need to address it in imbalanced dataset situation
    
    """
    def get_precision_recall_fm(self, pos_lable = '1'):
        """
        As the data is imbalanced we use weighted average for calculating the metrics
        output:
        return a dict for {'precision','recall','f1-score','accuracy'}
        """

        res = classification_report(self.y_true, self.y_pred, output_dict=True)
        res_pos = res[pos_lable]
        res_pos['accuaracy'] = res["accuracy"]

        return res_pos
    def get_brier(self):

        # y_score = \
        # (self.y_score - self.y_score.min()) / (self.y_score.max() - self.y_score.min())

        clf_score = sklearn.metrics.brier_score_loss(self.y_true, self.y_score, pos_label=self.y_max)

        return clf_score

    def get_roc_metrics(self):
        """
        calculate area under the ROC curve and average precision score (AProc) for positive class

        output:
        return a dictionary for both scores {'aucroc', 'auprc'}
        """
        precision, recall, threashold = precision_recall_curve(self.y_true, self.y_score)
        auc_p_r = roc_auc_score(y_true=self.y_true, y_score=self.y_score, average="weighted")
        # Since the possitive class is more important, and the data is imbalanced, this mettic may fits better to our need
        prauc = average_precision_score(self.y_true, self.y_score )

        return {'AUROC': auc_p_r, 'PRAUC': prauc}

    def get_all_metrics(self):
        #claculate the metrics
        precision_recall_fscore = self.get_precision_recall_fm()
        res = dict(precision_recall_fscore)

        roc = self.get_roc_metrics()
        brier = self.get_brier()

        #update the dictionary to add the roc and brier
        res.update(roc)
        res.pop('support')
        res['Brier'] = brier

        return {self.classifire_name: res}

    def get_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def plot_cm(self, file_address=None):
        cm = self.get_confusion_matrix()
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix for " + self.classifire_name)
        if file_address:
            plt.savefig(file_address)
        else:
            return plt
