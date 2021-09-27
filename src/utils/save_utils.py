import logging
import os
import time


class MimicSave :
    _instance = None
    output_directory = ''
    def __init__(self, ):
        if MimicSave._instance != None:
            raise Exception("This class is a singleton")
        else:
            MimicSave._instance = self

    @staticmethod
    def get_instance():
        if MimicSave._instance == None:
            MimicSave()
        return MimicSave._instance

    def create_get_output_dir(self, save_dir, is_test=False, k_fold = None):
        if is_test:
            self.directory = save_dir
            self.log_dir = self.directory + "/log"
            self.model_dir = self.directory + "/model"
            self.results_dir = self.directory + '/results'
        else:
            time_str = time.strftime("%m%d-%H-%M-%S")

            self.directory = save_dir + "/" + time_str
            os.mkdir(self.directory)
            if k_fold is None:
                self.create_save_directories(save_dir)
            else:
                for i in range(k_fold):
                    self.create_save_directories(f'{self.directory}/{i}')
        return self.directory

    def get_directory(self, fold_index = None):
        if fold_index is None:
            return self.directory
        return self.directory + '/'+ fold_index

    def get_log_directory(self, fold_index = None):
        if fold_index is None :
            return self.directory + "/log"
        return f'{self.directory}/{fold_index}/log'

    def get_model_directory(self,fold_index = None):
        if fold_index is None:
            return self.directory + 'model'
        return f'{self.directory}/{fold_index}/model'

    def get_results_directory(self,fold_index = None):
        if fold_index is None :
            return self.directory + '/results'
        return f'{self.directory}/{fold_index}/results'


    def create_save_directories(self, directory):

        log_dir = directory + "/log"
        model_dir = directory + "/model"
        results_dir = directory + '/results'

        dir_list = [directory, log_dir, model_dir,results_dir]
        output_directory = directory
        for dir in dir_list:
            try:
                os.mkdir(dir)
            except OSError as error:
                print(error)