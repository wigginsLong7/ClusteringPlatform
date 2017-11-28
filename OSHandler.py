import scipy.io
import OSEnum
import numpy as np


class OSHandler:
    def __init__(self):
        return

    @staticmethod
    def load_data(path, mode):
        if mode == OSEnum.MAT_FORMAT:
            data = scipy.io.loadmat(path)
            return data
        elif mode == OSEnum.TXT_FORMAT:
            data_list = []
            for line in open(path, 'rb'):
                line = line.decode('utf-8')
                m = line.find(',')
                n = line.find('\t')
                z = line.find(' ')
                sub_str = []
                if m != -1:
                    sub_str = line.split(',')
                elif n!= -1 :
                    sub_str = line.split('\t')
                elif z !=-1:
                    sub_str = line.split(' ')
                data_e = []
                for e in sub_str:
                    data_e.append(float(e))
                data_list.append(data_e)
            return np.mat(data_list)



