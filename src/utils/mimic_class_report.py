from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve, \
    brier_score_loss, precision_score, recall_score, f1_score, plot_roc_curve
import time
import os
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve


TIME_STR= time.strftime("%m%d-%H-%M-%S")
class ClassificationReport:
    weight_default = "witghted"


    def __init__(self, clf, y_true, y_pred,y_score, output_dir):
        self.clf = clf
        self.y_pred = y_pred
        self.y_true = y_true
        self.y_score = y_score
        self.y_max = max(y_true)
        self.output_dir = output_dir
        self.classifire_name = self.clf.__class__.__name__

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
        if not hasattr(self.clf, "decision_function"):
            y_score = self.y_score
        else:
            y_score = \
                (self.y_score - self.y_score.min()) / (self.y_score.max() - self.y_score.min())
        try:
            clf_score = brier_score_loss(self.y_true, y_score, pos_label=self.y_max)
        except:
            print("negative number issue")
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

    def get_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def get_all_metrics(self):
       #claculate the metrics
        precision_recall_fscore = self.get_precision_recall_fm()
        roc = self.get_roc_metrics()
        res = dict(precision_recall_fscore)
        brier = self.get_brier()
       #update them
        res.update(roc)
        res.pop('support')
        res['Brier'] = brier

        clf_name = self.clf.__class__.__name__
        return {clf_name: res}

    def plot_cm(self, X_test, file_address=None):
        plot_confusion_matrix(self.clf, X_test, self.y_true)

        plt.title("Confusion Matrix for " + self.classifire_name)
        if file_address:
            plt.savefig(file_address)
        else:
            return plt

    def plot_roc(self, X_test):
        plot_roc_curve(self.clf, X_test, self.y_true)
        plt.title("ROC Curve for " + self.classifire_name)
        file_name = self.get_file_name(self.classifire_name + "-roc-curve.png")
        plt.savefig(file_name)

    def plot_pr_curve(self, X_test):
        disp = plot_precision_recall_curve(self.clf, X_test, self.y_true)
        disp.ax_.set_title(self.classifire_name + ': 2-class Precision-Recall curve: '
                                                  'AP={0:0.2f}'.format(self.get_roc_metrics()["PRAUC"]))
        return disp

    # TODO need to be re implemented !! regarding
    def save_plots(self, X_test, output_dir):
        timestr = time.strftime("%m%d-%H%M%S")

        cm_plot = self  .plot_cm(X_test=X_test)
        file_name = self.get_file_name(self.classifire_name + "-confusion_matrix.png")
        cm_plot.savefig(file_name)
        roc_disp = self.plot_pr_curve(X_test=X_test)
        file_name = self.get_file_name(self.classifire_name + "-roc_prauc.png")
        plt.savefig(file_name)
        self.plot_roc(X_test)

    def get_file_name(self, file_name):
        try:
            os.mkdir(self.output_dir + "/" + TIME_STR)
        except OSError as error:
             pass
        return self.output_dir + "/" + TIME_STR + "/" + file_name

    def plot_calibration_curve(self, fig_index, X_train, X_test, y_train, y_test):
        """Plot calibration curve for est w/o and with calibration. """
        # Calibrated with isotonic calibration
        name = self.classifire_name
        est = self.clf
        isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

        # Calibrated with sigmoid calibration
        sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

        # Logistic regression with no calibration as baseline
        lr = LogisticRegression(C=1.)

        fig = plt.figure(fig_index, figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for clf, name in [(lr, 'Logistic'),
                          (est, name),
                          (isotonic, name + ' + Isotonic'),
                          (sigmoid, name + ' + Sigmoid')]:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                prob_pos = clf.predict_proba(X_test)[:, 1]
            else:  # use decision function
                prob_pos = clf.decision_function(X_test)
                prob_pos = \
                    (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

            clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_train.max())
            print("%s:" % name)
            print("\tBrier: %1.3f" % (clf_score))
            print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
            print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
            print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_test, prob_pos, n_bins=10)

            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                     label="%s (%1.3f)" % (name, clf_score))

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                     histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        file_name = self.get_file_name(self.classifire_name + "-calibration-plot.png")
        plt.savefig(file_name, dpi=150)


import matplotlib.pyplot as plt


class TablePlot():
    title = 'MIMIC-III Results'
    footer = time.strftime("%Y-%m-%d-%H:%M:%S")


    def __init__(self, clf_results, output_dir, float_format="%.3f"):
        self.df_clf_results = clf_results
        self.output_dir = output_dir
        self.float_format = float_format

    def draw_table(self):
        """
        @TODO Documnet me
        """
        columns_title = list(self.df_clf_results.columns)
        row_title = list(self.df_clf_results.index)
        np_table = self.df_clf_results.to_numpy()

        # Table data needs to be non-numeric text. Format the data
        cell_text = []
        for row in np_table:
            cell_text.append([f'{x:1.3f}' for x in row])
        # Get some lists of color specs for row and column headers
        rcolors = plt.cm.BuPu(np.full(len(row_title), 0.1))
        ccolors = plt.cm.BuPu(np.full(len(columns_title), 0.1))

        # Create the figure. Setting a small pad on tight_layout
        # seems to better regulate white space. Sometimes experimenting
        # with an explicit figsize here can produce better outcome.
        plt.figure(linewidth=2,
                   tight_layout={'pad': 1},
                   # figsize=(5,3)
                   )
        # Add a table at the bottom of the axes
        the_table = plt.table(cellText=cell_text,
                              rowLabels=row_title,
                              rowColours=rcolors,
                              rowLoc='right',
                              colColours=ccolors,
                              colLabels=columns_title,
                              loc='center')
        # Scaling is the only influence we have over top and bottom cell padding.
        # Make the rows taller (i.e., make cell y scale larger).
        the_table.scale(1, 1.5)
        # Hide axes
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Hide axes border
        plt.box(on=None)
        # Add title
        plt.suptitle(self.title)
        # Add footer
        plt.figtext(0.95, 0.05, self.footer, horizontalalignment='right', size=6, weight='light')
        # Force the figure to update, so backends center objects correctly within the figure.
        # Without plt.draw() here, the title will center on the axes and not the figure.
        plt.draw()

        # Create image. plt.savefig ignores figure edge and face colors, so map them.
        fig = plt.gcf()
        file_name = self.get_file_name("table_classifiers_results.png")
        plt.savefig(file_name,
                    # bbox='tight',
                    edgecolor=fig.get_edgecolor(),
                    facecolor=fig.get_facecolor(),
                    dpi=150
                    )
        print(row_title)

    def get_file_name(self, file_name):
        try:
            os.mkdir(self.output_dir + "/" + TIME_STR)
        except OSError as error:
            pass
        return self.output_dir + "/" + TIME_STR + "/" + file_name

    def print_table(self):
        print(self.clf_results.to_markdown())

    def save_to_latex(self):
        file_name = self.get_file_name("latex.tex")
        with open(file_name, 'w') as tf:
            tf.write(self.df_clf_results.to_latex(float_format=self.float_format))

    def save_to_excel(self):
        file_name = self.get_file_name("excel.xlsx")
        self.df_clf_results.to_excel(file_name, float_format=self.float_format)

    # df_table.style.apply(highlight_max)
    def highlight_max(s):
        '''
        highlight the maximum in a Series yellow.
        '''
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]
