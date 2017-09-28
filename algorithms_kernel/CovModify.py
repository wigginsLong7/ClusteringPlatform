import numpy as np
import types


class CovModify:
    def __init__(self):
        return

    @ staticmethod
    def throw_neg_eigenvalue(m):
        d, v = np.linalg.eig(m)
        a = np.eye(d.size)
        for j in range(d.size):
            if d[j] < 0:
                a[j][j] = 0
            else:
                a[j][j] = d[j]
        r = v*a*np.linalg.inv(v)
        return r

    @staticmethod
    def judge_single_matrix(m):
        if isinstance(m, type(np.mat([1, 1]))):
            try:
                inv_m = np.linalg.inv(m)
                return True
            except Exception as err:
                print(err)
                return False
        else:
            print("the format of mat is not correct")
            return False

    @staticmethod
    def cov_modify(m, t=1E-6):
        if not isinstance(m, type(np.mat([1, 1]))):
            print("the format of mat is not correct")
            return None
        s = m.shape
        non_neg_eigen_value_m = CovModify.throw_neg_eigenvalue(m)
        if not CovModify.judge_single_matrix(non_neg_eigen_value_m):
            non_neg_eigen_value_m += t*np.eye(s[0])
        else:
            pass
        return non_neg_eigen_value_m




