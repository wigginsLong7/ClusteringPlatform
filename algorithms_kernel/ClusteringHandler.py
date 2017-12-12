import numpy as np
import random
import sys
import math
from algorithms_kernel.CovModify import CovModify
import algorithms_kernel.AlgorithmsEnum as AlgorithmsEnum
import time
from sklearn.decomposition import PCA


# 抽象类, 最顶层聚类句柄
class ClusteringHandler(object):

    def __init__(self):
        self.Clustering_Handler_Type = 0

    @ staticmethod
    def check_data_and_k(source_data, k):
        if not isinstance(k, int):
            print('Error,the type of Cluster number is not integer')
            return False
        if not isinstance(source_data, type(np.mat([1, 1]))):
            print('Error,the type of source data is not matrix')
            return False
        if len(source_data) <= k:
            print("Error,The amount of data is not enough")
            return False
        if k < 1:
            print("Error, The value of k must be positive")
            return False
        s = source_data.shape
        data_num = s[0]
        data_dim = s[1]
        if data_num <= data_dim:
            print("Warning, The dimensionality of data is quite large compared with its amount")
        return True

    @staticmethod
    def _run(self):
        return


# 抽象类，基于中心点聚类
class ClusteringHandlerBOCP(ClusteringHandler):

    def __init__(self):
        super(ClusteringHandler, self).__init__()  # 使用super函数
        self.Clustering_Handler_Type = 1

    @staticmethod
    def K_Means(source_data, k, max_step=100, seed_num=1, stop_threshold=1E-3):

        '''
        Input:
        source_data: N x D source data matrix (N : sample size of data , D:  dimensionality of data )
        k:          the number of initial clusters
        max_step:   the upper bound of iteration times
        seed_num:  the seed value on controlling initial random state of k clusters
        stop_threshold: the changing ratio of likelihood value to judge convergence. 0.001 will be acceptable

        Output:
        centers:  k x D matrix, the position of center of all clusters (k: the number of clusters, D: the dimensionality
                  of data)
        label:    N x 1 vector, the mark label data point belongs to (N : sample size of data)
        '''

        if not ClusteringHandler.check_data_and_k(source_data, k):
            return False, [], []  #run_symbol, center, label
        s = source_data.shape
        data_num = s[0]
        data_dim = s[1]

        # random initial
        random.seed(seed_num)
        a = [i for i in range(data_num)]
        m = random.sample(a, k)

        #pre define
        centers = np.mat(np.zeros((k, data_dim)))
        data_id = np.mat(np.zeros((data_num, 1)))

        last_centers = np.mat(np.zeros((k, data_dim)))  # 保存上一次数值
        last_data_id = np.mat(np.zeros((data_num, 1)))

        for i in range(k):
            centers[i] = source_data[m[i]]
        sum_threshold = sys.float_info.max
        sum_map = np.mat(np.zeros((data_num, k)))# 不是 matrix, 纯粹记录用
        stop_step = 0

        # main_fun
        for i in range(max_step):
            for j in range(k):
                distance = source_data - np.tile(centers[j], (data_num, 1))
                distance_square = np.sum(np.multiply(distance, distance), axis=1)
                sum_map[:, j] = distance_square
            for n in range(data_num):
                m_index = np.where(sum_map[n] == sum_map[n].min())[1]
                data_id[n, 0] = random.sample(set(m_index), 1)[0]
            now_sum = 0.0
            for j in range(k):
                j_index = np.where(data_id == j)[0]
                if len(j_index) > 0:  # 非空类
                    centers[j] = source_data[j_index].mean(axis=0)
                    distance = source_data[j_index] - np.tile(centers[j], (len(j_index), 1))
                    distance_square = np.sum(np.multiply(distance, distance))
                    now_sum += distance_square
                else:  # 空类，应该返回上一次的东西，不是现在的计算的数值(权宜之计)
                    return True, last_centers, last_data_id
            last_centers = centers
            last_data_id = data_id
            if abs(now_sum - sum_threshold) / sum_threshold < stop_threshold:
                break
            sum_threshold = now_sum
            stop_step = i
        if stop_step == max_step - 1:
            print('K_means method does not coverage')
        return True, centers, data_id


# 抽象类，基于概率模型聚类
class ClusteringHandlerBODM(ClusteringHandler):

    def __init__(self):
        super(ClusteringHandler, self).__init__()
        self.Clustering_Handler_Type = 2


# 抽象类，基于密度估计聚类
class ClusteringHandlerBODE(ClusteringHandler):

    def __init__(self):
        super(ClusteringHandler, self).__init__()
        self.Clustering_Handler_Type = 3


class DBSCANClustering(ClusteringHandlerBODE):
    def __init__(self):
        super(ClusteringHandlerBODE, self).__init__()
        self.Clustering_Handler_Type = 6

    @staticmethod
    def distance(a, b):
        '''
           This takes the Euler distance as the measurement
        '''
        t = a - b
        return math.sqrt(np.sum(np.multiply(t, t)))

    @staticmethod
    def region_query(data, point_id, eps):

        s = data.shape
        data_num = s[0]
        seeds = []
        for i in range(data_num):
            if DBSCANClustering.distance(data[point_id], data[i]) < eps:
                seeds.append(i)
        return seeds

    @staticmethod
    def expand_cluster(data, cluster_result, point_id, cluster_id, eps, min_pts):

        seeds = DBSCANClustering.region_query(data, point_id, eps)
        if len(seeds) < min_pts:  # 不满足minPts条件的为噪声点
            cluster_result[point_id] = AlgorithmsEnum.DBSCAN_NOISE_POINT
            return False
        else:
            cluster_result[point_id] = cluster_id  # 划分到该簇
            for seed_id in seeds:
                cluster_result[seed_id] = cluster_id
            seed_index = 0
            while seed_index <= len(seeds):
                current_point = seeds[seed_index]
                query_results = DBSCANClustering.region_query(data, current_point, eps)
                if len(query_results) >= min_pts:
                    for e in query_results:
                        if cluster_result[e] == AlgorithmsEnum.DBSCAN_UNCLASSIFIED:
                            seeds.append(e)
                            cluster_result[e] = cluster_id
                        elif cluster_result[e] == AlgorithmsEnum.DBSCAN_NOISE_POINT:
                            cluster_result[e] = cluster_id
                seed_index += 1
            return True

    @staticmethod
    def clustering(data, eps, min_pts):
        '''

        Input：
        data : N x D source data matrix (N : sample size of data , D:  dimensionality of data )
        eps :  The threshold value of the neighborhood, the similarity value (distance at most case) which is less than
               the value is judged as true (density-reachable)
        min_pts: Not noise points must have at least min_pts points in their neighborhood

        output：
        cluster_id: N x 1 vector, the mark label data point belongs to (N : sample size of data)
        k: the number of centers

        '''

        if not isinstance(data, type(np.mat([1, 1]))):
            print('Error,the type of source data is not matrix')
            return False, [], -1
        cluster_id = 1
        s = data.shape
        data_sum = s[0]
        cluster_result = np.mat(np.zeros(data_sum, 1))
        for i in range(s[0]):
            cluster_result[i] = AlgorithmsEnum.DBSCAN_UNCLASSIFIED
        for point_id in range(data_sum):
            if cluster_result[point_id] == AlgorithmsEnum.DBSCAN_UNCLASSIFIED:
                if DBSCANClustering.expand_cluster(data, cluster_result, point_id, cluster_id, eps, min_pts):
                    cluster_id += 1
        return True, cluster_result, cluster_id - 1


# 抽象类，其他的聚类方法
class ClusteringHandlerBOO(ClusteringHandler):
    def __init__(self):
        super(ClusteringHandler, self).__init__()
        self.Clustering_Handler_Type = 4


class SpectralClustering(ClusteringHandlerBOO):
    def __init__(self):
        super(ClusteringHandlerBOO, self).__init__()
        self.Clustering_Handler_Type = 5

    @staticmethod
    def clustering_in_sklearn(source_data, k):

        return

    @staticmethod
    def build_similarity_graph(source_data):

        if not isinstance(source_data, type(np.mat([1, 1]))):
            print('Error,the type of source data is not matrix')
            return []
        s = source_data.shape
        g_map = np.mat(np.zeros((s[0], s[0])))
        for i in range(s[0]):
            for j in range(i + 1, s[0]):
                distance = np.sum((source_data[i] - source_data[j]) * (source_data[i] - source_data[j]).T)
                t_item_cov = np.cov(source_data[i], source_data[j])
                g_map[i, j] = math.exp(-1 * distance / t_item_cov)
                g_map[j, i] = g_map[i, j]
        return g_map

    @staticmethod
    def get_closest_k_element_on_similar_graph(w, args):

        sub_e = w
        k_value = args['args'][0]
        k_index_set = []
        while k_value > 0:
            max_index = np.where(sub_e == sub_e.max())[1]
            for e in max_index:
                k_index_set.append(e)
                k_value -= 1
                sub_e[0, e] = 0
                if k_value == 0:
                    break
        return k_index_set

    @staticmethod
    def get_similarity_matrix(similarity_graph, args):

        w = similarity_graph
        if not isinstance(similarity_graph, type(np.mat([1, 1]))):
            print('Error,the type of source data is not matrix')
            return []
        s = w.shape
        if args['mode'] == AlgorithmsEnum.SPECTRAL_CLUSTERING_ELISION:
            w[w < args['args'][0]] = 0
            w[w >= args['args'][0]] = 1
            return w
        elif args['mode'] == AlgorithmsEnum.SPECTRAL_CLUSTERING_KNN:  #KNN
            for i in range(s[0]):
                k_index_set = SpectralClustering.get_closest_k_element_on_similar_graph(w[i], args)
                w[i] = np.mat(np.zeros((1, s[0])))
                w[i, k_index_set] = 1
            return w
        elif args['mode'] == AlgorithmsEnum.SPECTRAL_CLUSTERING_MUTUAL_KNN: #互KNN
            temp_w = w
            w = np.mat(np.zeros((s[0], s[0])))
            for i in range(s[0]):
                k_index_set = SpectralClustering.get_closest_k_element_on_similar_graph(temp_w[i], args)
                for j in k_index_set:
                    sub_index_set = SpectralClustering.get_closest_k_element_on_similar_graph(temp_w[j], args)
                    if i in sub_index_set:
                        temp_w[i, j] = 1
                        temp_w[j, i] = 1
            w[temp_w == 1] = 1  # 数据不能重复
            return w
        else:
            return []

    @staticmethod
    def clustering(source_data, k, args):
        '''
        input:
        source_data:
        k:
        args:

        output:
        symbol:  whether the algorithm works
        centers:
        data_id:

        '''
        if not ClusteringHandler.check_data_and_k(source_data, k):
            return False
        s = source_data.shape
        s_g = SpectralClustering.build_similarity_graph(source_data)
        W = SpectralClustering.get_similarity_matrix(s_g, args)
        temp = np.sum(W, axis=1)
        D = np.mat(np.eye(s[0]))
        for j in range(s[0]):
            D[j, j] = temp[j, 0]
        if args['mode'] == AlgorithmsEnum.SPECTRAL_CLUSTERING_NOT_REGULAR_W:
            L = D - W
        elif args['mode'] == AlgorithmsEnum.SPECTRAL_CLUSTERING_RANDOM_W:
            L = np.linalg.inv(D)*(D - W)
        elif args['mode'] == AlgorithmsEnum.SPECTRAL_CLUSTERING_SYMMETRY_W:
            temp_d = np.linalg.inv(np.sqrt(D))
            L = temp_d * (D - W) * temp_d
        R = np.mat(np.zeros((s[0], k)))
        eig_val, eig_vector = np.linalg.eig(L)
        eig_val = eig_val.real
        index = np.argsort(-eig_val)
        for i in range(k):
            t = index[0, i]
            R[:, i] = eig_vector[:, t]
        if args['mode'] == AlgorithmsEnum.SPECTRAL_CLUSTERING_SYMMETRY_W:
            for j in range(s[0]):
                i_sum = np.sum(np.multiply(R[j], R[j]))
                i_sum = np.sqrt(i_sum)
                R[j] = R[j] / i_sum
        sy, centers, data_id = ClusteringHandlerBOCP.K_Means(R, k)
        return sy, centers, data_id


# 抽象类，基于概率模型聚类 GMM model
class ClusteringHandlerBOGMM(ClusteringHandlerBODM):

    def __init__(self):
        super(ClusteringHandlerBODM, self).__init__()
        self.Clustering_Handler_Type = 5

    @staticmethod
    def return_parameters_args():
        a = {'mode': AlgorithmsEnum.EM_MODE, 'max_step': 100, 'seed_num': 1, 'prior_threshold': 0.1, 'args': [0, 0, 0, 0]
             , 'now_step': -1, 'n_free_para': -1}
        return a

    @staticmethod
    def set_label(p):

        if not isinstance(p, type(np.mat([1, 1]))):
            print('Error,the type of probability is not matrix')
            return -1
        v_index = np.where(p == p.max())[1]
        if len(v_index) == 1:
            return v_index[0]
        elif len(v_index) > 1:
            return random.sample(set(v_index), 1)[0]
        else:
            return -1

    @staticmethod
    def initialization(data, k, seed_num=1, max_step=1):

        '''
        Input:
        data:   D x N  source matrix (D: the data dimensionality, N: sample size)
        k:      the Number K of GMM components.
        seed_num: the seed value on controlling initial random state

        output:
        priors:  1 x k array representing the prior probabilities of the  k GMM components.
        mu:      D x k array representing the centers of the k GMM components.
        sigma:   D x D x K array representing the covariance matrices of the k GMM components.  sigma[0] is D X D
        '''

        if seed_num < 1:
            seed_num = 1
        if max_step < 1:
            max_step = 1
        if k < 1:
            return False, [], [], []
        s = data.shape
        data_dim = s[0]
        sy, centers, data_id = ClusteringHandlerBOCP.K_Means(data.T, k, max_step, seed_num)
        if not sy:
            return False, [], [], []
        mu = centers.T
        priors = np.mat(np.ones((1, k)))
        sigma = []
        for i in range(k):
            sigma.append(np.mat(np.zeros((data_dim, data_dim))))
        for i in range(k):
            id_temp_index = np.where(data_id == i)[0]  # array
            priors[0, i] = len(id_temp_index)
            if priors[0, i] <= 0:
                priors[0, i] = 1
                sigma[i] = np.mat(np.zeros((data_dim, data_dim)))
            else:
                temp = (np.tile(data[:, id_temp_index], (1, 2))).T
                sigma[i] = np.mat(np.cov(temp.T))
            sigma[i] = CovModify.cov_modify(sigma[i])
        priors = priors / np.sum(priors)
        return True, priors, mu, sigma

    @staticmethod
    def get_free_parameter_num(k, data_dim):
        return (k - 1) + k * (data_dim + 0.5 * data_dim * (data_dim + 1))

    @staticmethod
    def calculate_gauss_probability_distribute(source_data, mu, sigma):

        '''
        Input:
        source_data:  D x N array representing N data points of D dimensions.
        mu:    D x 1 array representing the centers of the K GMM components.
        sigma: D x D  array representing the covariance matrices of the Gaussian component

        output:
        prob:  N x 1 array representing the probabilities for the  N data points.
        '''

        if not isinstance(source_data, type(np.mat([1, 1]))):
            print('Error,the type of source data is not matrix')
            return []
        if not isinstance(mu, type(np.mat([1, 1]))):
            print('Error,the type of mu is not matrix')
            return []
        if not isinstance(sigma, type(np.mat([1, 1]))):
            print('Error,the type of sigma is not matrix')
            return []
        t = sigma.shape
        if t[0] != t[1]:
            print('Error,something wrong in sigma')
            return []
        s = source_data.shape
        if s[0] > s[1]:
            print("Warning, The dimensionality of data is quite large compared with its amount")
        data_num = s[1]
        data_dim = s[0]
        data = source_data.T - np.tile(mu.T, (data_num, 1))
        prob = np.sum(np.multiply(data * sigma.I, data), axis=1)
        denominator = np.sqrt(pow((2 * math.pi), data_dim) * (abs(np.linalg.det(sigma))))
        if denominator == 0: # 除以0
            denominator += 1E-100
        prob = np.exp(-0.5 * prob) / denominator
        return prob

    @staticmethod
    def EM_Solution(source_data, k, args, centers_set, data_id_set, priors_set, sigma_set,likelihood_set):

        '''
        Input:
        source_data:  D x N  source matrix (D: the data dimensionality, N: sample size)
        k : the Number k of GMM components (category).
        max_step:   the upper bound of iteration times
        seed_num:   the seed value on controlling initial random state of k clusters

        output:
        run_flag: boolean variance, True indicates Success, False indicates Failed
        center:   k x D matrix, the position of center of all clusters (k: the number of clusters,
                  D: the dimensionality of data)

        label:   N x 1 vector, the mark label data point belongs to (N : sample size of data)

        nbStep:  Final times of iteration step

        likelihood:  return the mean of log(P(x|i));
        '''

        if not ClusteringHandler.check_data_and_k(source_data.T, k):
            return False, [], [], 0, 0 # run_symbol, center, label, step, log_like

        dynamic_k = k
        s = source_data.shape
        data_num = s[1]
        old_likelihood = 1
        n_step = 1
        likelihood_ratio_threshold = 1e-5
        args['n_free_para'] = ClusteringHandlerBOGMM.get_free_parameter_num(k, s[0])

        run_flag, priors, mu, sigma = ClusteringHandlerBOGMM.initialization(source_data, dynamic_k, args['seed_num'])
        if not run_flag:
            return False, [], [], 0, 0

        # 开辟内存
        label = np.mat(np.zeros((data_num, 1)))
        center = None
        probability_i_x = np.mat(np.zeros((data_num, dynamic_k)))

        while True:
            probability_i_x = ClusteringHandlerBOGMM.E_step_in_EM(source_data, dynamic_k, mu, sigma, priors, args)
            E = np.sum(probability_i_x, axis=0)
            if E[np.isnan(E)].size > 0:
                return False, center, label, n_step, old_likelihood
            sy, trim_set = ClusteringHandlerBOGMM.trimming_rule(dynamic_k, priors, mu, probability_i_x, sigma, E, args,
                                                                AlgorithmsEnum.FORCE_TRIMMING)
            if sy:
                dynamic_k, priors, mu, probability_i_x, sigma, E = ClusteringHandlerBOGMM.trim_components(dynamic_k,
                                                                 priors, mu, probability_i_x, sigma, E, trim_set)
            ClusteringHandlerBOGMM.M_step_in_EM(source_data, dynamic_k, mu, sigma, priors, probability_i_x)
            if args['mode'] != AlgorithmsEnum.EM_MODE:
                sy, trim_set = ClusteringHandlerBOGMM.trimming_rule(dynamic_k, priors, mu, probability_i_x, sigma, E,
                                                                     args, AlgorithmsEnum.SOFT_TRIMMING)
                if sy:
                    dynamic_k, priors, mu, probability_i_x, sigma, E = ClusteringHandlerBOGMM.trim_components(dynamic_k,
                                                                         priors, mu, probability_i_x, sigma, E, trim_set)
            center = mu.T
            for i in range(data_num):
                val = ClusteringHandlerBOGMM.set_label(probability_i_x[i])
                if val > -1:
                    label[i, 0] = val
            centers_set.append(center)
            data_id_set.append(label)
            sigma_set.append(sigma)
            priors_set.append(priors)

            sy, val = ClusteringHandlerBOGMM.stop_condition(source_data, dynamic_k, mu, sigma, priors, old_likelihood,
                                                            likelihood_ratio_threshold, n_step - args['max_step'])
            if val != -1:
                old_likelihood = val
            likelihood_set.append(old_likelihood)
            if not sy:
                break
            n_step += 1
            args['now_step'] = n_step
            time.sleep(0.5)
        if n_step >= args['max_step']:
            print("EM solution does not converge")
        return True, center, label, n_step, old_likelihood

    @staticmethod
    def modify_pix_on_lyya(priors, probability_x_i, args):

        s = probability_x_i.shape
        t = priors.shape
        probability_i_x_temp_no_eta = np.multiply(np.tile(priors, (s[0], 1)), probability_x_i)
        power_index = 1.0 + (1.0 / float(args['args'][0]))
        probability_i_x_temp = np.power(probability_i_x_temp_no_eta, power_index)
        for i in range(s[0]):
            if probability_i_x_temp[i][np.isinf(probability_i_x_temp[i])].size > 0 and \
                            probability_i_x_temp_no_eta[i][np.isinf(probability_i_x_temp_no_eta[i])].size == 0: # power_index 引起的
                temp = np.isinf(probability_i_x_temp[i])
                sub_set = np.where(temp == True)
                if len(sub_set) == 1:
                    probability_i_x_temp[i] = np.mat(np.zeros((1, t[1])))
                    probability_i_x_temp[i, sub_set[0]] = 1
                elif len(sub_set) > 1:
                    max_value = probability_i_x_temp_no_eta[i].max()
                    log_value = np.log10(probability_i_x_temp_no_eta[i])
                    temp_value = log_value - np.mat(np.tile(np.log10(max_value), (1, t[1])))
                    temp_value = np.power(temp_value, power_index)
                    temp_value = np.power(10, temp_value)
                    probability_i_x_temp[i] = temp_value
            if np.sum(abs(probability_i_x_temp[i])) == 0 and np.sum(abs(probability_i_x_temp_no_eta[i])) != 0:
                max_value = probability_i_x_temp_no_eta[i].max()
                log_value = np.log10(probability_i_x_temp_no_eta[i])
                temp_value = log_value - np.mat(np.tile(np.log10(max_value), (1, t[1])))
                temp_value = np.power(temp_value, power_index)
                temp_value = np.power(10, temp_value)
                probability_i_x_temp[i] = temp_value
        return probability_i_x_temp

    @staticmethod
    def modify_pix_on_byy(probability_i_x, probability_i_x_temp):

        s = probability_i_x.shape
        k = s[1]
        probability_i_x_temp[probability_i_x_temp <= 0] = 1E-100
        probability_i_x_temp_log = np.log(probability_i_x_temp)
        temp = np.sum(np.multiply(probability_i_x, probability_i_x_temp_log), axis=1)
        probability_i_x_factor = probability_i_x_temp_log - np.tile(temp, (1, k))
        probability_i_x = probability_i_x + np.multiply(probability_i_x, probability_i_x_factor)
        return True

    @staticmethod
    def modify_pix_on_rpcl(probability_i_x, args):

        s = probability_i_x.shape
        temp_pix = np.mat(np.zeros((s[0], s[1])))
        if args['mode'] == AlgorithmsEnum.HardCutEM_MODE:
            for i in range(s[0]):
                max_index = np.where(probability_i_x[i] == probability_i_x[i].max())[1]
                if len(max_index) >= 1:
                    f_index = random.sample(set(max_index), 1)[0]
                    temp_pix[i, f_index] = 1
                else:
                    return False
            sum_pix = np.sum(temp_pix, axis=1)
            probability_i_x = temp_pix / np.tile(sum_pix, (1, s[1]))
            return True
        elif args['mode'] == AlgorithmsEnum.RPCL_MODE:
            for i in range(s[0]):
                max_index = np.where(probability_i_x[i] == probability_i_x[i].max())[1]
                if len(max_index) >= 1:
                    f_index = random.sample(set(max_index), 1)[0]
                    temp_pix[i] = probability_i_x[i]
                    temp_pix[i, f_index] = 0
                    max_index2 = np.where(temp_pix[i] == probability_i_x[i].max())[1]
                    if len(max_index2) >= 1:
                        f_index_2 = random.sample(set(max_index2), 1)[0]
                        temp_pix[i, f_index_2] = -1 * args['args'][0]
                    else:
                        return False
                    temp_pix[i, f_index] = 1
                else:
                    return False
            sum_pix = np.sum(temp_pix, axis=1)
            probability_i_x = temp_pix / np.tile(sum_pix, (1, s[1]))
            return True
        else:
            return False

    @staticmethod
    def calculate_component_sigma(sigma):

        k = len(sigma)
        sigma_label = np.ones((k, 1))  # array
        whole_sum = 0.0
        true_num = k
        gaussian_sigma_value = np.zeros((k, 1))
        for i in range(k):
            gaussian_sigma_value[i, 0] = np.linalg.det(sigma[i])
            if gaussian_sigma_value[i, 0] <= 0:
                sigma_label[i, 0] = 0
                true_num -= 1
            else:
                whole_sum += gaussian_sigma_value[i, 0]
        if true_num <= 0:
            print('All components has incorrect sigma')
            return []
        total_mean = whole_sum / true_num
        for i in range(k):
            if gaussian_sigma_value[i, 0] < total_mean and sigma_label[i, 0] == 1:
                sigma_label[i, 0] = gaussian_sigma_value[i, 0] / total_mean
        return sigma_label

    @staticmethod
    def trim_components(dynamic_k, priors, mu, probability_i_x, sigma, E, trim_value_set):

        set_or_value = True
        if isinstance(trim_value_set, int):
            set_or_value = False
        elif isinstance(trim_value_set, np.ndarray):
            set_or_value = True
        else:
            print('trim_value_set is wrong type')
            return dynamic_k, priors, mu, probability_i_x, sigma, E
        if (set_or_value and len(trim_value_set) < 0) or (not set_or_value and trim_value_set < 0):
            return dynamic_k, priors, mu, probability_i_x, sigma, E
        E = np.delete(E, trim_value_set, axis=1)
        probability_i_x = np.delete(probability_i_x, trim_value_set, axis=1)
        mu = np.delete(mu, trim_value_set, axis=1)
        priors = np.delete(priors, trim_value_set, axis=1)
        if set_or_value:
            for e in trim_value_set:
                del sigma[e]
            new_k = dynamic_k - len(trim_value_set)
        else:
            del sigma[trim_value_set]
            new_k = dynamic_k - 1
        return new_k, priors, mu, probability_i_x, sigma, E

    @staticmethod
    def trimming_rule(dynamic_k, priors, mu, probability_i_x, sigma, E, args, mode):

        if mode == AlgorithmsEnum.FORCE_TRIMMING:
            zero_value_index = np.where(E <= 0)[1]
            if len(zero_value_index) > 0:
                if args['mode'] == AlgorithmsEnum.EM_MODE:
                    E[:, zero_value_index] = 1E-100
                    return False, []
                else:
                    return True, zero_value_index
            else:
                return False, []
        elif mode == AlgorithmsEnum.SOFT_TRIMMING:
            small_gaussian_components_label = ClusteringHandlerBOGMM.calculate_component_sigma(sigma)
            if not len(small_gaussian_components_label):
                return False, []
            min_sigma = 1E+100
            min_index = -1
            for i in range(dynamic_k):
                if priors[0, i] < args['prior_threshold'] and small_gaussian_components_label[i, 0] < 1:
                    if np.linalg.det(sigma[i]) < min_sigma:
                        min_sigma = np.linalg.det(sigma[i])
                        min_index = i
            if min_index > -1:
                return True, min_index
            else:
                return False, []
        else:
            return False, []

    @staticmethod
    def E_step_in_EM(source_data, k, mu, sigma, priors, args):

        s = source_data.shape
        data_num = s[1]
        probability_i_x = np.mat(np.zeros((data_num, k)))
        probability_i_x_temp = np.mat(np.zeros((data_num, k)))
        probability_x_i = np.mat(np.zeros((data_num, k)))
        for i in range(k):
            temp = ClusteringHandlerBOGMM.calculate_gauss_probability_distribute(source_data, np.mat(mu[:, i]), sigma[i])
            if len(temp):
                probability_x_i[:, i] = temp  # 有可能除以0
            else:
                return []
        if args['mode'] == AlgorithmsEnum.LAGRANGE_YING_YANG_ALTERNATION_MODE:
            probability_i_x_temp = ClusteringHandlerBOGMM.modify_pix_on_lyya(priors, probability_x_i, args)
        else:
            probability_i_x_temp = np.multiply(np.tile(priors, (data_num, 1)), probability_x_i)
        probability_i_x = probability_i_x_temp / (np.tile(np.sum(probability_i_x_temp, axis=1), (1, k)))
        t = np.where(np.sum(probability_i_x_temp, axis=1) == 0)[0]
        if len(t) > 0:
            for i2 in range(len(t)):
                for j2 in range(k):
                    probability_i_x[t[i2, 0], j2] = 0
        if args['mode'] == AlgorithmsEnum.RPCL_MODE or args['mode'] == AlgorithmsEnum.HardCutEM_MODE:
            ClusteringHandlerBOGMM.modify_pix_on_rpcl(probability_i_x, args)
        elif args['mode'] == AlgorithmsEnum.BYY_TWO_ALTERNATION_STEP_MODE:
            ClusteringHandlerBOGMM.modify_pix_on_byy(probability_i_x, probability_i_x_temp)
        return probability_i_x

    @staticmethod
    def M_step_in_EM(source_data, k, mu, sigma, priors, probability_i_x):

        s = source_data.shape
        data_num = s[1]
        data_dim = s[0]
        E = np.sum(probability_i_x, axis=0)
        for i in range(k):
            priors[0, i] = E[0, i] / data_num
            mu[:, i] = source_data * probability_i_x[:, i] / E[0, i]
            data_temp = source_data - np.tile(mu[:, i], (1, data_num))
            sigma[i] = np.multiply(np.tile(probability_i_x[:, i].T, (data_dim, 1)), data_temp)*data_temp.T / E[0, i]
            if sigma[i][np.isnan(sigma[i])].size > 0 or sigma[i][np.isinf(sigma[i])].size > 0:
                return False
            sigma[i] = CovModify.cov_modify(sigma[i])
        return True

    @staticmethod
    def stop_condition(source_data, k, mu, sigma, priors, old_likelihood, likelihood_threshold, step_gap):

        s = source_data.shape
        data_num = s[1]
        probability_x_i = np.mat(np.zeros((data_num, k)))
        for i in range(k):
            temp = ClusteringHandlerBOGMM.calculate_gauss_probability_distribute(source_data, np.mat(mu[:, i]), sigma[i])
            if len(temp):
                probability_x_i[:, i] = temp  # 有可能除以0
            else:
                return False, -1
        F = probability_x_i * priors.T
        F[F <= 0] = 1E-100
        now_likelihood = np.log(F).mean(axis=0)[0, 0]
        if abs((now_likelihood / old_likelihood) - 1) < likelihood_threshold or step_gap >= 0:
            return False, now_likelihood
        return True, now_likelihood

