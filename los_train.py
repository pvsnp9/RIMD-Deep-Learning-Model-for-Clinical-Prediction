import torch
import numpy as np
import pickle
import logging
import json


from torch import nn
from src.utils.mimic_iii_decay_data import MIMICDecayData
from src.model.mimi_decay_los_model import MIMICLosDecayModel
from src.model.mimic_lstm_model import MIMICLSTMLosModel
from src.model.mimic_grud_model import MIMICGRUDLosModel
from src.model.mimic_gru_model import MIMICGRULosModel
from src.utils.save_utils import MimicSave
from src.utils.mimic_args import args


np.random.seed(1048)
torch.manual_seed(1048)
torch.cuda.manual_seed(1048)

SAVE_DIR = 'mimic/los'

class Train:
    def __init__(self, args, data_object):
        self.data_object = data_object
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.set_decay_params()
        if args['model_type'] == 'RIMDecay':
            self.model = MIMICLosDecayModel(self.args).to(self.device)
        if args['model_type'] == 'LSTM':
            self.model = MIMICLSTMLosModel(self.args).to(self.device)
        if args['model_type'] == 'GRU':
            self.model = MIMICGRULosModel(self.args).to(self.device)
        if args['model_type'] == 'GRUD':
            self.model = MIMICGRUDLosModel(self.args).to(self.device)
        self.cell_types = ['LSTM', 'GRU']
        self.loss = nn.MSELoss()
        self.out_dir = MimicSave.get_instance().create_get_output_dir(SAVE_DIR)

    def config_logger(self):
        # Save the args used in this experiment
        with open(f'{self.out_dir}/_experiment_args.txt','w') as f:
            json.dump(self.args,f)

        #config logging
        logging.basicConfig(filename=self.out_dir + '/' + 'log.txt', format='%(message)s', level=logging.DEBUG)
        # Adding log to console as well
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logging.Formatter('*\t%(message)s'))
        logging.getLogger().addHandler(consoleHandler)
        logging.getLogger('matplotlib.font_manager').disabled = True
        logging.info("Run Description: LOS task")
        # logging.info("Log file created at " + datetime.now().strftime("%Y/%m/%d  %H:%M:%S"))
        logging.info("Directory: {0}".format(self.out_dir))

        # startTime = datetime.now()
        # logging.info('Start time: ' + str(startTime))

        return logging

    def set_decay_params(self):
        # self.data_object = MIMICDecayData(args['batch_size'], 24, args['input_file_path'])
        self.args['input_size'] = self.data_object.input_size
        self.args['static_features'] = self.data_object.statics_size
        self.args['mask_size'] = self.data_object.input_size
        self.args['delta_size'] = self.data_object.input_size
        self.args['hidden_size'] = self.data_object.input_size
        self.args['comm_value_size'] = self.data_object.input_size

    def train(self):
        train_loader, val_loader, test_loader = self.data_object.data_loader()
        logger = self.config_logger()
        logger.info(f'Model Arch: \n { self.model }')
        logger.info(
            f"\n Training, Validating, and Testing: {self.args['model_type']} model with {self.args['rnn_cell']} cell ")
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=0.9)
        self.model.train()
        for epoch in range(self.args['epochs']):
            epoch_loss = 0.0
            iteration = 0
            logger.info("###################### Training ################")
            for x, static, x_mean, y in train_loader:
                if self.args['model_type'] == 'RIMDecay' or self.args['model_type'] == 'GRUD':
                    static = static.to(self.device)
                    x_mean = x_mean.to(self.device)
                    x_mask = x[:,1,:,:].to(self.device)
                    delta = x[:,2,:,:].to(self.device)
                    x_last_ob = x[:, 3, :, :].to(self.device)
                    x = x[:, 0, :, :].to(self.device)

                    y = y.float().to(self.device)
                    output = self.model(x, static, x_mask, delta, x_last_ob, x_mean)
                    output = torch.squeeze(output)
                    loss = self.loss(output, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    iteration += 1
                    epoch_loss += loss.item()
                else:
                    static = static.to(self.device)
                    x = x[:, 0, :, :].to(self.device)

                    y = y.float().to(self.device)
                    output = self.model(x, static)
                    output = torch.squeeze(output)
                    loss = self.loss(output, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    iteration += 1
                    epoch_loss += loss.item()

            val_loss = self.eval(val_loader)
            test_loss = self.eval(test_loader)

            logger.info(f'EPOCH [{epoch+1}], training loss: {epoch_loss/iteration}, val loss: {val_loss}, test_loss: {test_loss}')
    

    def eval(self, data_loader):
        self.model.eval()
        iter_loss = 0.0
        iteration = 0
        with torch.no_grad():
            for x, statics, x_mean, y in data_loader:
                if self.args['model_type'] == 'RIMDecay' or self.args['model_type'] == 'GRUD':
                    static = statics.to(self.device)
                    x_mask = x[:, 1, :, :].to(self.device)
                    delta = x[:, 2, :, :].to(self.device)
                    x_mean = x_mean.to(self.device)
                    x_last_ob = x[:, 3, :, :].to(self.device)
                    x = x[:, 0, :, :].to(self.device)

                    y = y.to(self.device)
                    output = self.model(x, static, x_mask, delta, x_last_ob, x_mean)
                    output = torch.squeeze(output)
                    loss = self.loss(output, y)
                    iter_loss += loss.item()
                    iteration += 1
                else:
                    static = statics.to(self.device)
                    x = x[:, 0, :, :].to(self.device)

                    y = y.to(self.device)
                    output = self.model(x, static)
                    output = torch.squeeze(output)
                    loss = self.loss(output, y)
                    iter_loss += loss.item()
                    iteration += 1


        return iter_loss /iteration         

if __name__ == '__main__':
    decay_data_object = MIMICDecayData(args['batch_size'], 24, args['decay_input_file_path'])
    model_types = ['RIMDecay', 'GRUD', 'LSTM', 'GRU']
    cell_types = ['LSTM', 'GRU']
    for model_type in model_types:
        if model_type == 'RIMDecay':
            args['model_type'] = 'RIMDecay'
            for cell_type in cell_types:
                args['rnn_cell'] = cell_type
                trainer = Train(args, decay_data_object)
                trainer.train()
                del trainer
        elif model_type == 'GRUD':
            args['model_type'] = 'GRUD'
            trainer = Train(args, decay_data_object)
            trainer.train()
            del trainer
        elif model_type == 'LSTM':
            args['model_type'] = 'LSTM'
            trainer = Train(args, decay_data_object)
            trainer.train()
            del trainer
        elif model_type == 'GRU':
            args['model_type'] = 'GRU'
            trainer = Train(args, decay_data_object)
            trainer.train()
            del trainer
        else:
            print("could not find options to train")