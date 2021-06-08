from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve
import time
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve


class ClassificationReport:
    weight_default = "witghted"
    timestr = time.strftime("%m%d-%H-%M-%S")
    def __init__(self, clf, y_true, y_pred, output_dir):
        self.clf = clf
        self.y_pred = y_pred
        self.y_true = y_true
        self.output_dir=output_dir
        self.classifire_name = self.clf.__class__.__name__

    def get_precision_recall_fm(self, average="weighted"):
        """
        As the data is imbalanced we use weighted average for calculating the metrics
        output:
        return a dict for {'precision','recall','f1-score','support'
        """
        # TODO extract class wise accouracy if needed, add a key_extract filter to do so
        res = classification_report(self.y_true, self.y_pred, output_dict=True)
        return res['weighted avg']

    def get_roc_metrics(self, X_test):
        """
        calculate area under the ROC curve and average precision score (AProc) for positive class

        output:
        return a dictionary for both scores {'aucroc', 'auprc'}
        """
        y_score = self.clf.decision_function(X_test)

        precision, recall, threashold = precision_recall_curve(self.y_true, y_score)
        auc_p_r = roc_auc_score(y_true=self.y_true,y_score=y_score, average="weighted")
        # Since the possitive class is more important, and the data is imbalanced, this mettic may fits better to our need
        prauc = average_precision_score(self.y_true, y_score,)

        return {'aucroc': auc_p_r, 'prauc': prauc}

    def get_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def get_all_metrics(self, X_test):
        precision_recall_fscore = self.get_precision_recall_fm()
        roc = self.get_roc_metrics(X_test)
        res = dict(precision_recall_fscore)
        res.update(roc)
        res.pop('support')
        clf_name = self.clf.__class__.__name__
        return {clf_name:res}

    def plot_cm(self, X_test, file_address=None):
        plot_confusion_matrix(self.clf, X_test, self.y_true)

        plt.title("Confusion Matrix for "+ self.classifire_name)
        if file_address:

            plt.savefig(file_address)
        else:
            return plt

    def plot_roc_curve(self, X_test):

        disp = plot_precision_recall_curve(self.clf, X_test, self.y_true)
        disp.ax_.set_title(self.classifire_name+': 2-class Precision-Recall curve: '
                           'AP={0:0.2f}'.format(self.get_roc_metrics(X_test)["prauc"]))
        return disp



    def save_plots(self,X_test,output_dir):
        timestr = time.strftime("%m%d-%H%M%S")

        cm_plot = self.plot_cm(X_test=X_test)
        file_name = self.get_file_name(self.classifire_name+"-confusion_matrix.png")
        cm_plot.savefig(file_name)
        roc_disp = self.plot_roc_curve(X_test=X_test)
        file_name  = self.get_file_name(self.classifire_name+"-roc_prauc.png")
        plt.savefig(file_name)

    def get_file_name(self, file_name):
        return self.output_dir + "/" + self.timestr + "-" + file_name


import matplotlib.pyplot as plt
class TablePlot():

    title = 'MIMIC-III Results'
    footer = time.strftime("%Y-%m-%d-%H:%M:%S")
    timestr = time.strftime("%m%d-%H-%M-%S")
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

    def get_file_name (self,file_name):
        return self.output_dir+"/"+self.timestr+"-"+file_name

    def print_table(self):
        print(self.clf_results.to_markdown())

    def save_to_latex(self):
        file_name = self.get_file_name("latex.tex")
        with open(file_name, 'w') as tf:
            tf.write(self.df_clf_results.to_latex(float_format=self.float_format))


    def save_to_excel(self):
        file_name = self.get_file_name("excel.xlsx")
        self.df_clf_results.to_excel(file_name,float_format=self.float_format)



    # df_table.style.apply(highlight_max)
    def highlight_max(s):
        '''
        highlight the maximum in a Series yellow.
        '''
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]


