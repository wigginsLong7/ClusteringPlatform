import threading
from algorithms_kernel.ClusteringHandler import *
from OSHandler import OSHandler
from algorithms_kernel.DrawHandler import *
import OSEnum
import time


class ThreadTest:
    def __init__(self):
        self.args = ClusteringHandlerBOGMM.return_parameters_args()
        self.args['mode'] = AlgorithmsEnum.HardCutEM_MODE
        self.args['prior_threshold'] = 0.1
        self.args['args1'] = 2
        return

    def thread_start(self):
        g_th = threading.Thread(target=self.run, args=())
        g_th.setDaemon(True)  # 守护线程
        g_th.start()
        return

    def run(self):
        c = OSHandler()
        s = c.load_data('E:\\DataSet(11_26)\\44.txt', OSEnum.TXT_FORMAT)
        sy, center, label, n_step, old_likelihood = ClusteringHandlerBOGMM.EM_Solution(s.T, 10, self.args)
        print(center)
        DrawHandler.draw_graph(s, center, label)

t = ThreadTest()
t.thread_start()
for i in range(100):
    print(t.args['now_step'] / t.args['max_step'])
    print('haha')
    print(t.args['now_step'])
    time.sleep(2)