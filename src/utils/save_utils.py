import os
import time


class MimicSave :
    _instance = None
    directory = ''
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


    def create_get_output_dir(self, output_dir):
        time_str= time.strftime("%m%d-%H-%M-%S")
        directory =  output_dir + "/" + time_str
        self.directory = directory
        try:
            os.mkdir(directory)
        except OSError as error:
            pass
        return directory

    def get_directory(self):
        return self.directory