import os
import time


class MimicSave :
    _instance = None
    output_directory = ''
    def __init__(self):
        if MimicSave._instance != None:
            raise Exception("This class is a singleton")
        else:
            MimicSave._instance = self

    @staticmethod
    def get_instance():
        if MimicSave._instance == None:
            MimicSave()
        return MimicSave._instance


    def create_get_output_dir(self, save_dir):
        time_str= time.strftime("%m%d-%H-%M-%S")

        directory = save_dir + "/" + time_str
        self.log_dir =  directory + "/log"
        self.model_dir =  directory + "/model"


        dir_list = [directory, self.log_dir, self.model_dir ]
        self.output_directory = directory
        for dir in dir_list:
            try:
                os.mkdir(dir)
            except OSError as error:
                pass
        return directory

    def get_directory(self):
        return self.output_directory

    def get_log_directory(self):
        return self.log_dir

    def get_model_directory(self):
        return self.model_dir
