from numpy.core.fromnumeric import argmax
import torch
import pickle
import numpy as np
from src.model.mimic_model import MIMICModel
from src.model.mimic_lstm_model import MIMICLSTMModel
from src.model.mimic_gru_model import MIMICGRUModel
from src.utils.mimic_evaluation import MIMICReport
from src.utils.mimic_iii_data import MIMICIIIData
from src.utils.data_prep import MortalityDataPrep
from src.model.mimic_decay_model import MIMICDecayModel
from src.model.mimic_decay_with_cb_loss import MIMICDecayCBLossModel
from src.model.mimic_grud_model import MIMICGRUDModel
from src.utils.mimic_iii_decay_data import MIMICDecayData
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve
import time

from src.utils.save_utils import MimicSave

'''
!!!! Important !!!!!!!
Input arguments like model_type and cell type are crucial.
'''


# torch.manual_seed(10)
# np.random.seed(10)
# torch.cuda.manual_seed(10)


class TrainModels:
    def __init__(self, args=None, data_object=None, logger=None):

        self.save_dir = MimicSave.get_instance().get_model_directory()
        self.log_dir = MimicSave.get_instance().get_log_directory()

        self.reports = {}
        self.logger = logger
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if args is not None:
            self.args = args
            self.data_object = None
            self.model = None
            self.cell_types = ['LSTM', 'GRU']
            self.data_object = data_object
            if args['model_type'] == 'RIMDecay' or args['model_type'] == 'GRUD' :
                self.set_decay_params()
            else:
                self.set_default_params()
            self.set_model_name()

            self.logger.info(args)
        else:
            self.logger.info('-------------- Model initiated for prediction ---------------------')

    def set_decay_params(self):
        # self.data_object = MIMICDecayData(args['batch_size'], 24, args['input_file_path'])
        self.args['input_size'] = self.data_object.input_size
        self.args['static_features'] = self.data_object.statics_size
        self.args['mask_size'] = self.data_object.input_size
        self.args['delta_size'] = self.data_object.input_size
        self.args['hidden_size'] = self.data_object.input_size
        self.args['comm_value_size'] = self.data_object.input_size

    def set_default_params(self):
        # self.data_object = MIMICIIIData(args['batch_size'], 24, args['input_file_path'], args['mask'])
        self.args['input_size'] = self.data_object.input_size
        self.args['static_features'] = self.data_object.statics_size
        if self.args['model_type'] == 'LSTM':
            self.cell_types.pop(1)
        elif self.args['model_type'] == 'GRU':
            self.cell_types.pop(0)

    def tune_train(self):
        """
        Objective function for tuning the models, at this time PRAUC has been selected which can be modified latter based
        on the results and discussions
        """

        res = self.train()
        if res == 0:
            return 0

        y_truth, y_pred, y_score = res[self.model_name]
        report = MIMICReport(self.model_name, y_truth, y_pred, y_score, './figures')

        return report.get_roc_metrics()['PRAUC']

    def set_model_name(self):
        if self.args['model_type'] in ['LSTM', 'GRU']:
            self.model_name = self.args['model_type']
        else:
            self.model_name = f"{self.args['model_type']}_{self.args['rnn_cell']}"

    """
    based on the is_cbloss we will call the desired function but need to change this code using a proper method 
    like factory method or something 
    """
    def train(self):
        if self.args['is_cbloss']:
            return self.train_cb_loss()
        else:
            return self.train_normal()


    def train_normal(self):

        ctr = 0
        start_epochs = 0
        loss_stats = []
        acc = []
        train_acc = []
        test_acc = []
        valid_f1 = []
        train_f1 = []
        # test_f1 = []
        t_f1 = 0
        max_val_accuracy = 0.0
        max_val_f1 = 0.0
        epochs_no_improve = 0
        best_epoch_number = 0
        max_no_improvement = self.args['max_no_improvement']
        improvement_threshold = self.args['improvement_threshold']

        # for cell in self.cell_types:
        train_loader, val_loader, test_loader = self.data_object.data_loader()
        # self.args['rnn_cell'] = cell
        if self.args['model_type'] == 'RIMDecay':
            self.model = MIMICDecayModel(self.args).to(self.device)
        if self.args['model_type'] == 'GRUD':
            self.model = MIMICGRUDModel(self.args).to(self.device)
        elif self.args['model_type'] == 'RIM':
            self.model = MIMICModel(self.args).to(self.device)
        elif self.args['model_type'] == 'LSTM':
            self.model = MIMICLSTMModel(self.args).to(self.device)
        else:
            self.model = MIMICGRUModel(self.args).to(self.device)

        self.logger.info(f'Model Arch: \n {self.model}')
        self.logger.info(
            f"Training, Validating, and Testing: {self.args['model_type']} model with {self.args['rnn_cell']} cell ")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        for epoch in range(start_epochs, self.args['epochs']):
            self.logger.info(f"******************** {self.model_name} *************************************")
            self.logger.info(
                f'--- EPOCH: {epoch + 1} --------------------------------- B-F1: {format(max_val_f1 * 100, ".2f")} '
                f'from EPOCH: {best_epoch_number + 1},   Test-F1 : {format(t_f1 * 100, ".2f")}')

            start = time.time()
            epoch_loss = 0.0
            iter_ctr = 0.0
            t_accuracy = 0
            norm = 0
            y_pred = []
            y_truth = []
            self.model.train()
            if self.args['model_type'] == 'RIMDecay' or self.args['model_type'] == 'GRUD':
                for x, static, x_mean, y in train_loader:
                    iter_ctr += 1

                    static = static.to(self.device)
                    x_mask = x[:, 1, :, :].to(self.device)
                    delta = x[:, 2, :, :].to(self.device)
                    x_mean = x_mean.to(self.device)
                    x_last_ob = x[:, 3, :, :].to(self.device)
                    x = x[:, 0, :, :].to(self.device)

                    y = y.to(self.device)

                    output, l = self.model(x, static, x_mask, delta, x_last_ob, x_mean, y)

                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                    norm += self.model.grad_norm()

                    epoch_loss += l.item()

                    predictions = torch.round(output)

                    correct = predictions.view(-1) == y.long()
                    t_accuracy += correct.sum().item()
                    y_truth.append(y) #y.cpu().detach().numpy()
                    y_pred.append(predictions) #predictions.view(-1).cpu().detach().numpy()
                    ctr += 1
            else:
                for x, statics, y in train_loader:
                    iter_ctr += 1

                    x = x.to(self.device)
                    statics = statics.to(self.device)
                    y = y.to(self.device)

                    output, l = self.model(x, statics, y)

                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                    norm += self.model.grad_norm()

                    epoch_loss += l.item()
                    predictions = torch.round(output)
                    correct = predictions.view(-1) == y.long()
                    t_accuracy += correct.sum().item()
                    y_truth.append(y)
                    y_pred.append(predictions )
                    ctr += 1
            y_truth = torch.cat(y_truth)
            y_pred = torch.cat(y_pred)
            y_truth = y_truth.cpu().detach().numpy()
            y_pred = y_pred.view(-1).cpu().detach().numpy()
            self.logger.info(f"----- time spent for training  : {time.time() - start} -----")
            start = time.time()
            train_f1_report = classification_report(y_truth, y_pred, output_dict=True)
            self.logger.info(f" time for train set report : {time.time() - start} ")

            start = time.time()
            validation_accuracy, val_f1 = self.eval(val_loader)
            self.logger.info(f"eval time for val set : {time.time() - start } ")

            # TODO Commented as we are not using it and in iteration it takes almost 3 seconds !, we might be able
            # to calculate the eval for validation set concurrently to reduce the time spent for each epoch

            test_accuracy, t_f1 = self.eval(test_loader)

            self.logger.info(
                f'epoch loss: {format(epoch_loss, ".3f")}, taining accuracy: {format(t_accuracy / len(train_loader.dataset) * 100, ".2f")},'
                f' validation accuracy: {format(validation_accuracy * 100, ".2f")}, '
                f'Val-F1-score: {format(val_f1 * 100, ".2f")}, Test-F1 : {format(t_f1 * 100, ".2f")} ')

            try:
                train_f1.append((epoch, train_f1_report['1']['f1-score']))
            except:
                self.logger.info("Error : there is no sample with class label '1' !!!!")
            valid_f1.append((epoch, val_f1))
            # test_f1.append((epoch, t_f1))

            loss_stats.append((ctr, epoch_loss / iter_ctr))
            acc.append((epoch, (validation_accuracy)))
            train_acc.append((epoch, (t_accuracy / self.data_object.train_instances)))
            # test_acc.append((epoch, (test_accuracy)))
            # TODO  Sensetive loss function
            # early stopping code

            improve = val_f1 - max_val_f1
            if improve > improvement_threshold:
                epochs_no_improve = 0
                max_val_f1 = val_f1
                best_epoch_number = epoch
                # TODO chose the best model !?
                # TODO set the best_epoch and report it (epoch with best results)
                self.logger.info(f"New Best F1 Score at epoch {epoch} : {format( max_val_f1 * 100, '.2f')}")
                best_model_state = {
                    'net': self.model.state_dict(),
                    'epochs': epoch,
                    'args': self.args
                }
            else:
                epochs_no_improve += 1

            if epochs_no_improve == max_no_improvement:
                self.logger.info(f"Early Stopping @ EPOCH:[{epoch}]")
                break

        self.logger.info("saving the models state...")
        self.model_saved_fname = f"{self.save_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_model.pt"
        with open(self.model_saved_fname, 'wb') as f:
            torch.save(best_model_state, f)

        if not self.args['is_tuning']:
            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_Epoch_Loss.pickle", 'wb') as f:
                pickle.dump(loss_stats, f)
            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_Dev_Accuracy.pickle",
                      'wb') as f:
                pickle.dump(acc, f)

            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_Train_Accuracy.pickle",
                      'wb') as f:
                pickle.dump(train_acc, f)

            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_Test_Accuracy.pickle",
                      'wb') as f:
                pickle.dump(test_acc, f)

            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_Train_f1.pickle", 'wb') as f:
                pickle.dump(train_f1, f)

            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_Dev_f1.pickle", 'wb') as f:
                pickle.dump(valid_f1, f)

            # with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_Test_f1.pickle", 'wb') as f:
            #     pickle.dump(test_f1, f)

        # right after training do the test step

        self.reports.update({self.model_name:
                                 self.test(self.model_saved_fname, self.data_object.get_test_data())})
        return self.reports

    def eval(self, data_loader):
        accuracy = 0
        self.model.eval()
        y_truth = []
        y_pred = []
        with torch.no_grad():
            if self.args['model_type'] == 'RIMDecay' or self.args['model_type'] == 'GRUD':
                # TODO there is a bug in number of samples for test and val data sets
                for x, static, x_mean, y in data_loader:
                    static = static.to(self.device)
                    x_mask = x[:, 1, :, :].to(self.device)
                    delta = x[:, 2, :, :].to(self.device)
                    x_mean = x_mean.to(self.device)
                    x_last_ob = x[:, 3, :, :].to(self.device)
                    x = x[:, 0, :, :].to(self.device)

                    y = y.to(self.device)

                    predictions = self.model(x, static, x_mask, delta, x_last_ob, x_mean)
                    probs = torch.round(torch.sigmoid(predictions))
                    correct = probs.view(-1) == y
                    accuracy += correct.sum().item()
                    # add them to a list to calculate f1 score later on
                    y_truth.extend(y.cpu().detach().numpy())
                    y_pred.extend(probs.view(-1).cpu().detach().numpy())


            else:
                for x, statics, y in data_loader:
                    x = x.to(self.device)
                    statics = statics.to(self.device)
                    y = y.to(self.device)

                    predictions = self.model(x, statics)

                    probs = torch.round(torch.sigmoid(predictions))
                    correct = probs.view(-1) == y
                    accuracy += correct.sum().item()

                    # add them to a list to calculate f1 score later on
                    y_truth.extend(y.cpu().detach().numpy())
                    y_pred.extend(probs.view(-1).cpu().detach().numpy())

        # compute the f-1 measure
        report = classification_report(y_truth, y_pred, output_dict=True, zero_division=0)
        try:
            f1_score = report['1']['f1-score']
        except Exception as e:
            f1_score = 0
            print(report)
            print(e)
        if f1_score <= 0.01:
            print(f1_score)
            pass

        accuracy /= len(data_loader.dataset)
        return accuracy, f1_score

    def test(self, model_path, test_data):
        checkpoint = torch.load(model_path)
        args = checkpoint['args']
        if args['model_type'] == 'RIMDecay':
            model = MIMICDecayModel(args).to(self.device)
        elif args['model_type'] == 'RIM':
            model = MIMICModel(args).to(self.device)
        elif args['model_type'] == 'LSTM':
            model = MIMICLSTMModel(args).to(self.device)
        elif args['model_type'] == 'GRU':
            model = MIMICGRUModel(args).to(self.device)
        else:
            raise Exception('No model type found: {}'.format(model_path))

        self.args = args
        self.set_model_name()

        model.load_state_dict(checkpoint['net'])
        self.logger.info(f'Loaded model arch: \n {model}')

        if args['model_type'] == 'RIMDecay' or self.args['model_type'] == 'GRUD':
            x, static, x_mean, y = test_data
            static = static.to(self.device)
            x_mask = x[:, 1, :, :].to(self.device)
            delta = x[:, 2, :, :].to(self.device)
            x_mean = x_mean.to(self.device)
            x_last_ob = x[:, 3, :, :].to(self.device)
            x = x[:, 0, :, :].to(self.device)
            y = y.to(self.device)
            predictions = model(x, static, x_mask, delta, x_last_ob, x_mean)
            probs = torch.round(torch.sigmoid(predictions))
        else:
            x, statics, y = test_data
            x = x.to(self.device)
            statics = statics.to(self.device)
            y = y.to(self.device)
            predictions = model(x, statics)
            probs = torch.round(torch.sigmoid(predictions))

        gt = y.cpu().detach().numpy()
        pt = probs.view(-1).cpu().detach().numpy()
        y_score = torch.sigmoid(predictions).view(-1).cpu().detach().numpy()

        return gt, pt, y_score

    def train_cb_loss(self):

        ctr = 0
        start_epochs = 0
        loss_stats = []
        acc = []
        train_acc = []
        test_acc = []
        valid_f1 = []
        train_f1 = []
        test_f1 = []
        best_epoch_number =0
        max_val_accuracy = 0.0
        epochs_no_improve = 0
        max_no_improvement = self.args['max_no_improvement']
        improvement_threshold = self.args['improvement_threshold']
        max_val_f1= 0
        # for cell in self.cell_types:
        train_loader, val_loader, test_loader = self.data_object.data_loader()
        # self.args['rnn_cell'] = cell
        if self.args['model_type'] == 'RIMDecay':

            self.model = MIMICDecayCBLossModel(self.args).to(self.device)
        elif self.args['model_type'] == 'RIM':
            self.model = MIMICModel(self.args).to(self.device)
        else:
            raise Exception(f"No models found for {self.args['model_type']}")
        # elif self.args['model_type'] == 'LSTM':
        #     self.model = MIMICLSTMModel(self.args).to(self.device)
        # else:
        #     self.model = MIMICGRUModel(self.args).to(self.device)

        self.logger.info(f'Model Arch: \n {self.model}')
        self.logger.info(
            f"Training, Validating, and Testing: {self.args['model_type']} model with {self.args['rnn_cell']} cell ")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        for epoch in range(start_epochs, self.args['epochs']):
            start =  time.time()
            self.logger.info(f"*****************{self.model_name} -- CBLoss ***************************")
            self.logger.info(
                f'--- EPOCH: {epoch + 1} ---------------------------------  B-F1: {  format(max_val_f1 * 100, ".2f") } from EPOCH: {best_epoch_number +1} ')
            epoch_loss = 0.0
            iter_ctr = 0.0
            t_accuracy = 0
            norm = 0
            start = time.time()
            y_pred = []
            y_truth = []
            self.model.train()
            if self.args['model_type'] == 'RIMDecay':
                for x, static, x_mean, y in train_loader:
                    iter_ctr += 1

                    static = static.to(self.device)
                    x_mask = x[:, 1, :, :].to(self.device)
                    delta = x[:, 2, :, :].to(self.device)
                    x_mean = x_mean.to(self.device)
                    x_last_ob = x[:, 3, :, :].to(self.device)
                    x = x[:, 0, :, :].to(self.device)

                    y = y.to(self.device)

                    predictions, l = self.model(x, static, x_mask, delta, x_last_ob, x_mean, y)

                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                    norm += self.model.grad_norm()

                    output = torch.argmax(predictions, dim=1)

                    epoch_loss += l.item()
                    correct = y.long() == output
                    t_accuracy += correct.sum().item()
                    y_truth.extend(y.cpu().detach().numpy())
                    y_pred.extend(output.cpu().detach().numpy())

                    ctr += 1
            else:
                for x, statics, y in train_loader:
                    iter_ctr += 1

                    x = x.to(self.device)
                    statics = statics.to(self.device)
                    y = y.to(self.device)

                    predictions, l = self.model(x, statics, y)
                    output = torch.argmax(predictions, dim=1)

                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                    norm += self.model.grad_norm()

                    epoch_loss += l.item()
                    correct = output == y.long()
                    t_accuracy += correct.sum().item()

                    y_truth.extend(y.cpu().detach().numpy())
                    y_pred.extend(output.cpu().detach().numpy())

                    ctr += 1
            self.logger.info(f"time spent for training : {time.time() - start }")
            train_f1_report = classification_report(y_truth, y_pred, output_dict=True)
            validation_accuracy, val_f1 = self.eval_cb_loss(val_loader)
            test_accuracy, t_f1 = self.eval_cb_loss(test_loader)

            self.logger.info(
                f'epoch loss: {format(epoch_loss, ".3f")}, taining accuracy: {format(t_accuracy / len(train_loader.dataset) * 100, ".2f")},'
                f' validation accuracy: {format(validation_accuracy * 100, ".2f")}, '
                f'Val-F1-score: {format(val_f1 * 100, ".2f")}, Test-F1 : {format(t_f1 * 100 , ".2f")} ')

            # TODO - Warning!!! the below code has been added to stop bad parameters which cause large loss and continuing
            # Training is not desired !
            if epoch_loss > 1000:
                self.logger.warn("Un acceptable loss")
                return 0
            # append f1 score
            try:
                train_f1.append((epoch, train_f1_report['1']['f1-score']))
            except:
                self.logger.info("Error : there is no sample with class label '1' !!!!")
            valid_f1.append((epoch, val_f1))
            test_f1.append((epoch, t_f1))

            loss_stats.append((ctr, epoch_loss / iter_ctr))
            acc.append((epoch, (validation_accuracy)))
            train_acc.append((epoch, (t_accuracy / self.data_object.train_instances)))
            test_acc.append((epoch, (test_accuracy)))
            # TODO  Sensetive loss function
            # early stopping code

            improve = val_f1 - max_val_f1
            if improve > improvement_threshold:
                epochs_no_improve = 0
                max_val_f1 = val_f1
                best_epoch_number =  epoch
                # TODO chose the best model !?
                # TODO set the best_epoch and report it (epoch with best results)
                self.logger.info(f"New Best F1 Score at epoch {epoch} : {format(max_val_f1 * 100, '.2f')}")
                best_model_state = {
                    'net': self.model.state_dict(),
                    'epochs': epoch,
                    'args': self.args
                }
            else:
                epochs_no_improve += 1

            if epochs_no_improve == max_no_improvement:
                self.logger.info(f"Early Stopping @ EPOCH:[{epoch}]")
                break
                # TODO change the metric for choosing the best model to return !
        self.logger.info("saving the models state...")
        self.model_saved_fname = f"{self.save_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_cbloss_model.pt"
        with open(self.model_saved_fname, 'wb') as f:
            torch.save(best_model_state, f)

        if not self.args['is_tuning']:
            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_cbloss_Epoch_Loss.pickle", 'wb') as f:
                pickle.dump(loss_stats, f)
            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_cbloss_Dev_Accuracy.pickle",
                      'wb') as f:
                pickle.dump(acc, f)

            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_cbloss_Train_Accuracy.pickle",
                      'wb') as f:
                pickle.dump(train_acc, f)

            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_cbloss_Test_Accuracy.pickle",
                      'wb') as f:
                pickle.dump(test_acc, f)

            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_cbloss_Train_f1.pickle", 'wb') as f:
                pickle.dump(train_f1, f)

            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_cbloss_Dev_f1.pickle", 'wb') as f:
                pickle.dump(valid_f1, f)

            with open(f"{self.log_dir}/{self.args['model_type']}_{self.args['rnn_cell']}_cbloss_Test_f1.pickle", 'wb') as f:
                pickle.dump(test_f1, f)

        # right after training do the test step

        self.reports.update({self.model_name:
                                 self.test_cb_loss(self.model_saved_fname, self.data_object.get_test_data())})
        return self.reports

    def eval_cb_loss(self, data_loader):
        accuracy = 0
        self.model.eval()
        y_truth = []
        y_pred = []
        with torch.no_grad():
            if self.args['model_type'] == 'RIMDecay':
                # TODO there is a bug in number of samples for test and val data sets
                for x, static, x_mean, y in data_loader:
                    static = static.to(self.device)
                    x_mask = x[:, 1, :, :].to(self.device)
                    delta = x[:, 2, :, :].to(self.device)
                    x_mean = x_mean.to(self.device)
                    x_last_ob = x[:, 3, :, :].to(self.device)
                    x = x[:, 0, :, :].to(self.device)

                    y = y.to(self.device)

                    predictions = self.model(x, static, x_mask, delta, x_last_ob, x_mean)
                    output = torch.argmax(predictions, dim=1)
                    correct = output == y.long()
                    accuracy += correct.sum().item()
                    # add them to a list to calculate f1 score later on
                    y_truth.extend(y.cpu().detach().numpy())
                    y_pred.extend(output.cpu().detach().numpy())


            else:
                for x, statics, y in data_loader:
                    x = x.to(self.device)
                    statics = statics.to(self.device)
                    y = y.to(self.device)

                    predictions = self.model(x, statics)

                    output = torch.argmax(predictions,dim=1)
                    correct = output == y.long()
                    accuracy += correct.sum().item()

                    # add them to a list to calculate f1 score later on
                    y_truth.extend(y.cpu().detach().numpy())
                    y_pred.extend(output.cpu().detach().numpy())

        # compute the f-1 measure
        report = classification_report(y_truth, y_pred, output_dict=True, zero_division=0)
        try:
            f1_score = report['1']['f1-score']
        except Exception as e:
            f1_score = 0
            print(report)
            print(e)
        if f1_score <= 0.01:
            print(f1_score)
            pass

        accuracy /= len(data_loader.dataset)
        return accuracy, f1_score

    def test_cb_loss(self, model_path, test_data):
        checkpoint = torch.load(model_path)
        args = checkpoint['args']
        if args['model_type'] == 'RIMDecay':
            model = MIMICDecayCBLossModel(args).to(self.device)
        elif args['model_type'] == 'RIM':
            model = MIMICModel(args).to(self.device)
        elif args['model_type'] == 'LSTM':
            model = MIMICLSTMModel(args).to(self.device)
        elif args['model_type'] == 'GRU':
            model = MIMICGRUModel(args).to(self.device)
        else:
            raise Exception('No model type found: {}'.format(model_path))

        self.args = args
        self.set_model_name()

        model.load_state_dict(checkpoint['net'])
        self.logger.info(f'Testing with CB-Loss function model')
        self.logger.info(f'Loaded model arch: \n {model}')

        if args['model_type'] == 'RIMDecay':
            x, static, x_mean, y = test_data
            static = static.to(self.device)
            x_mask = x[:, 1, :, :].to(self.device)
            delta = x[:, 2, :, :].to(self.device)
            x_mean = x_mean.to(self.device)
            x_last_ob = x[:, 3, :, :].to(self.device)
            x = x[:, 0, :, :].to(self.device)
            y = y.to(self.device)
            predictions = model(x, static, x_mask, delta, x_last_ob, x_mean)
            probs = torch.argmax(predictions, dim=1)
        else:
            x, statics, y = test_data
            x = x.to(self.device)
            statics = statics.to(self.device)
            y = y.to(self.device)
            predictions = model(x, statics)
            probs = torch.argmax(predictions,dim=1)

        start  =  time.time()
        gt = y.cpu().detach().numpy()
        pt = probs.cpu().detach().numpy()

        y_score , indexes = torch.max(torch.softmax(predictions, dim=1),dim=1)
        y_score =  y_score.cpu().detach().numpy()
        print(f"time to detach arrays { time.time() - start }")
        return gt, pt, y_score

# def make_report(model):
#     # data = MIMICIIIData(64, 24, args['input_file_path'])  # MIMICDecayData(64, 24, args['input_file_path'])
#     test_data = data_object.get_test_data()
#     if .model_saved_fname !=None:
#         model_path = self.model_saved_fname
#     else:
#         model_path = f'./mimic/models/{model}_model.pt'
#     trainer = TrainModels()
#     y_truth, y_pred, y_score = trainer.test(model_path, test_data)
#
#     clf = MIMICReport(model, y_truth, y_pred, y_score, './figures')
#
#     results = clf.get_all_metrics()
#     cf_m = clf.get_confusion_matrix()
#     print(results)
#     print(cf_m)
#     print(classification_report(y_truth, y_pred))


# class TrainMlModels():
#     def __init__(self, args):


# train_model(model, args['epochs'], data)
