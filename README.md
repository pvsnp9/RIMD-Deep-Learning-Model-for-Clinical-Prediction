# RIMD-Deep Learning Model for Clinical Prediction

Electronic health records (EHRs) are sparse, noisy, and private, with variable patient vital measurements and stay lengths. Deep learning models are the current state of the art in many machine learning domains; however, the EHR data is not a suitable training input for most of them. In this paper, we introduce RIMD, a novel deep learning model that consists of a decay mechanism, modular recurrent
networks, and a custom loss function that learns minor classes in EHR. The decay mechanism learns from patterns in sparse data. The modular network allows multiple recurrent networks to pick only relevant input based on the attention score at a given timestamp. Finally, the custom CB loss function
is responsible for learning minor classes based on samples provided during training. This novel model is used to evaluate predictions for early mortality identification, length of stay, and acute respiratory failure on MIMIC-III dataset. Experiment results indicate that the proposed models outperform similar
models in F1-Score, AUROC, and PRAUC scores.

#####Note: Latest code is at <a href="https://github.com/pvsnp9/RIMD-Deep-Learning-Model-for-Clinical-Prediction/tree/metrics_baselines" target="_blank">Branch</a>



#### Paper
<a href="https://www.dropbox.com/s/6jgddfxpn7xite3/RIMD-Deep-Learning-Model-for-Clinical-Prediction.pdf" target="_blank">RIMD-Deep Learning Model for Clinical Prediction</a>

