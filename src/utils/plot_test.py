import matplotlib.pyplot
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from classification_report import ClassificationReport, TablePlot


from sklearn.metrics import roc_auc_score


output = '../../output/plots'


classifires = [ SVC(random_state=3), LogisticRegression(solver="liblinear", random_state=0)]
X, y = make_classification(random_state=0, n_classes=2,n_samples=1000, weights=[.90],flip_y=0.4)
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=3)

reports = []
for clf in classifires:
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    report = ClassificationReport(clf,y_test, y_pred, output_dir=output)
    reports.append(report)

clf_result = {}
index = 1
for report in reports:
    res_ = report.get_all_metrics(X_test=X_test)
    clf_result.update(res_)
    report.plot_calibration_curve(fig_index=index,X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test)

#convert the results to a dataframe
df_table = pd.DataFrame(clf_result)
df_table = df_table.T

s = df_table.style
s.highlight_max(axis=1)
df_table.style.apply(s)


print(df_table.to_latex(float_format="%.3f"))

table_report = TablePlot(df_table, output_dir=output)
table_report.draw_table()
table_report.save_to_excel()
table_report.save_to_latex()

print(df_table.to_markdown())


## plot diagrams for confusion Matrix and Precision-Recall curve

for report in reports:
        report.save_plots(X_test=X_test, output_dir=output)

