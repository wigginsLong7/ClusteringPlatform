from algorithms_kernel.ClusteringHandler import *
from OSHandler import OSHandler
from algorithms_kernel.DrawHandler import *
import OSEnum
import time

c = OSHandler()
s = c.load_data('E:\\DataSet(11_26)\\44.txt', OSEnum.TXT_FORMAT)
s2 = c.load_data('E:\\DataSet(11_26)\\s_set\\s3.txt', OSEnum.TXT_FORMAT)
#sy,center,label = ClusteringHandlerBOCP.K_Means(s, 4, 100, 10)
args = ClusteringHandlerBOGMM.return_parameters_args()
args['mode'] = AlgorithmsEnum.LAGRANGE_YING_YANG_ALTERNATION_MODE
args['prior_threshold'] = 0.2
args['args1'] = 2
args['seed_num'] = 1
start = time.clock()
sy, center, label, n_step, old_likelihood = ClusteringHandlerBOGMM.EM_Solution(s.T, 11, args)
end = time.clock()
print('Running time: %s Seconds'% (end-start))
print(center)
print(n_step)

DrawHandler.draw_graph(s, center, label)