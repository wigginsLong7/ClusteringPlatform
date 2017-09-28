import numpy as np
import matplotlib.pyplot as plt


class DrawHandler:
    def __init__(self):
        return

    @staticmethod
    def judge_matrix(m):
        if not isinstance(m, type(np.mat([1, 1]))):
            print('Error,the type of source data is not matrix')
            return False
        s = m.shape
        if s[1] != 2:
            print('Error,the data is not 2D')
            return False
        return True

    @staticmethod
    def draw_graph(data, center, label):
        if not DrawHandler.judge_matrix(data) or not DrawHandler.judge_matrix(center):
            return False
        plt.scatter([data[:, 0]], [data[:, 1]], s=20, c='r', marker='o')
        plt.scatter([center[:, 0]], [center[:, 1]], s=50, c='b', marker='o')
        plt.legend(['data', 'center'])
        plt.show()
        return True

    @staticmethod
    def draw_original_data_point(data, label, mode=False):
        if not DrawHandler.judge_matrix(data):
            return False
        if not mode:
            plt.scatter([data[:, 0]], [data[:, 1]], s=20, c='r', marker='o')
            plt.legend('data')
            plt.show()
        return True

    @staticmethod
    def draw_center_point(center):
        if not DrawHandler.judge_matrix(center):
            return False
        plt.scatter([center[:, 0]], [center[:, 1]], s=50, c='b', marker='o')
        plt.legend('center')
        plt.show()
        return True
