from algorithms_kernel.ClusteringHandler import *
from OSHandler import OSHandler
from algorithms_kernel.DrawHandler import *
import OSEnum
import time
import os
import sys

c = OSHandler()
pth = sys.path[0]
s = c.load_data(pth+'/44.txt', OSEnum.TXT_FORMAT)
s2 = c.load_data(pth+'/s3.txt', OSEnum.TXT_FORMAT)
#sy,center,label = ClusteringHandlerBOCP.K_Means(s, 4, 100, 10)
args = ClusteringHandlerBOGMM.return_parameters_args()
args['mode'] = AlgorithmsEnum.LAGRANGE_YING_YANG_ALTERNATION_MODE
args['prior_threshold'] = 0.2
args['args1'] = 2
args['seed_num'] = 1
start = time.clock()
c_set = []
id_set = []
s_set = []
p_set = []
sy, center, label, n_step, old_likelihood = ClusteringHandlerBOGMM.EM_Solution(s.T, 11, args, c_set, id_set, p_set, s_set)
end = time.clock()
print('Running time: %s Seconds'% (end-start))
print(center)
print(n_step)

DrawHandler.draw_graph(s, center, label)