import tkinter as tk
from tkinter import *
from tkinter import ttk


class AlgorithmPage(tk.Toplevel):
    def __init__(self, master, initial_para=None):
        super().__init__()
        self.title('设置算法')
        self.root = master
        self.geometry('%dx%d' % (450, 400))
        self.MODES = [
            ("probabilistic model clustering", "0"),
            ("density clustering", "1"),
            ("centroid clustering", "2"),
            ("hierarchical clustering", "3"),
            ("other clustering", "4")
        ]
        self.v = StringVar()
        self.v.set("0")  # initialize
        self.combobox_containers = [None, None, None, None, None]

        self.Algorithm_info = None
        self.combobox_args = [StringVar(), StringVar(), StringVar(), StringVar(), StringVar()]
        #self.algorithm_label_string_args = [["arg1", "arg2", "arg3"], [], [], [], []]
        self.algorithm_label_string_args = [[StringVar(), StringVar(), StringVar()], [StringVar(), StringVar()], [StringVar(), StringVar()], [StringVar(), StringVar()], [StringVar(), StringVar()]]
        self.algorithm_entry_string_args = [[StringVar(), StringVar(), StringVar()], [StringVar(), StringVar()], [StringVar(), StringVar()], [StringVar(), StringVar()], [StringVar(), StringVar()]]
        self.algorithm_label_containers = [[None, None, None], [None, None], [None, None], [None, None], [None, None]]
        self.algorithm_entry_containers = [[None, None, None], [None, None], [None, None], [None, None], [None, None]]
        self.setup_ui()
        self.algorithm_info_initial(initial_para)

    def algorithm_info_initial(self, d):
        self.Algorithm_info = {}
        if not d:
            self.Algorithm_info["clustering_type"] = 0
            self.Algorithm_info["algorithm_name"] = "HardCutEM"
            self.Algorithm_info["parameters"] = []
            self.Algorithm_info["current_index"] = 1
        else:
            for i in d:
                self.Algorithm_info[i] = d[i]
        m = self.Algorithm_info["clustering_type"]
        self.v.set(str(m))
        for i in range(len(self.combobox_containers)):
            if i == m:
                self.combobox_containers[i]['state'] = NORMAL
            else:
                self.combobox_containers[i]['state'] = DISABLED
        self.set_label_entry_visible(m)
        self.combobox_containers[m].current(self.Algorithm_info["current_index"])
        count = 0
        for j in self.Algorithm_info["parameters"]:
            self.algorithm_entry_string_args[m][count].set(j)

    def setup_ui(self):

        self.combobox_containers[0] = ttk.Combobox(self, textvariable=self.combobox_args[0], state='readonly', width=10)
        self.combobox_containers[0]['values'] = ('EM', 'HardCutEM', 'RPCL', 'BYY', 'LYYA')
        self.combobox_containers[0].grid(row=2, column=5, pady=10, columnspan=3)
        self.combobox_containers[0].current(1)
        for i in range(3):
            self.algorithm_label_containers[0][i] = Label(self, textvariable=self.algorithm_label_string_args[0][i], width=10)
            self.algorithm_label_containers[0][i].grid(row=3, column=2*i+1)
            self.algorithm_entry_containers[0][i] = Entry(self, textvariable=self.algorithm_entry_string_args[0][i], width=5)
            self.algorithm_entry_containers[0][i].grid(row=3, column=2*i+2)
            self.algorithm_label_string_args[0][i].set("args:")

        self.combobox_containers[1] = ttk.Combobox(self, textvariable=self.combobox_args[1], state='readonly', width=10)
        self.combobox_containers[1]['values'] = ('DBSCAN', 'OTHER')
        self.combobox_containers[1].grid(row=4, column=5, pady=10, columnspan=3)
        self.combobox_containers[1].current(0)

        self.combobox_containers[2] = ttk.Combobox(self, textvariable=self.combobox_args[2], state='readonly', width=10)
        self.combobox_containers[2]['values'] = ('Kmeans', 'Kcentroid')
        self.combobox_containers[2].grid(row=6, column=5, pady=10, columnspan=3)
        self.combobox_containers[2].current(0)

        self.combobox_containers[3] = ttk.Combobox(self, textvariable=self.combobox_args[3], state='readonly')
        self.combobox_containers[3]['values'] = ('tradition', 'other')
        self.combobox_containers[3].grid(row=8, column=5, pady=10, columnspan=3)
        self.combobox_containers[3].current(0)

        self.combobox_containers[4] = ttk.Combobox(self, textvariable=self.combobox_args[4], state='readonly', width=10)
        self.combobox_containers[4]['values'] = ('spectral', 'other')
        self.combobox_containers[4].grid(row=10, column=5, pady=10, columnspan=3)
        self.combobox_containers[4].current(0)

        for j in range(4):
            for i in range(2):
                self.algorithm_label_containers[j + 1][i] = Label(self, textvariable=self.algorithm_label_string_args[0][i],
                                                                  width=10)
                self.algorithm_label_containers[j + 1][i].grid(row=2 * j + 5, column=2 * i + 1)
                self.algorithm_entry_containers[j + 1][i] = Entry(self, textvariable=self.algorithm_entry_string_args[0][i],
                                                                  width=5)
                self.algorithm_entry_containers[j + 1][i].grid(row=2 * j + 5, column=2 * i + 2)
                self.algorithm_label_string_args[j + 1][i].set("args:")

        tk.Button(self, text="取消", command=self.cancel).grid(row=12, column=1, pady=10)
        tk.Button(self, text="确定", command=self.ok).grid(row=12, column=2, pady=10)

        st_row = 2
        for text, mode in self.MODES:
            b = Radiobutton(self, text=text,
                            variable=self.v, value=mode, command=self.radio_button_event)
            b.grid(row=st_row, column=1, pady=10, columnspan=4)
            st_row += 2

        for j in range(5):
            self.combobox_containers[j].bind('<<ComboboxSelected>>', lambda _: self.combobox_content_change(j))

    def combobox_content_change(self, *args):
        print("combobox"+str(args[0]))
        m = int(self.v.get())
        self.set_label_entry_visible(m)

    def set_label_entry_visible(self, clustering_type):
        for i in range(5):
            for j in self.algorithm_label_containers[i]:
                j.grid_forget()
            for z in self.algorithm_entry_containers[i]:
                z.grid_forget()
        if clustering_type == 0:
                if self.combobox_args[0].get() == "RPCL" or self.combobox_args[0].get() == "LYYA":
                    self.algorithm_label_containers[0][0].grid(row=3, column=1)
                    self.algorithm_entry_containers[0][0].grid(row=3, column=2)
        else:
            for i in range(2):
                self.algorithm_label_containers[clustering_type][i].grid(row=2 * clustering_type + 3, column=2 * i + 1)
                self.algorithm_entry_containers[clustering_type][i] .grid(row=2 * clustering_type + 3, column=2 * i + 2)
                self.algorithm_label_string_args[clustering_type][i].set("args:")

    def radio_button_event(self):
        m = int(self.v.get())
        for i in range(len(self.combobox_containers)):
            if i == m:
                self.combobox_containers[i]['state'] = NORMAL
            else:
                self.combobox_containers[i]['state'] = DISABLED
        self.set_label_entry_visible(m)

    def ok(self):
        self.Algorithm_info = {}
        para = []
        for i in self.algorithm_entry_string_args[int(self.v.get())]:
            if i.get() != "":
                para.append(i.get())
        self.Algorithm_info["clustering_type"] = int(self.v.get())
        self.Algorithm_info["algorithm_name"] = self.combobox_args[int(self.v.get())].get()
        self.Algorithm_info["parameters"] = para
        count = 0
        for j in self.combobox_containers[int(self.v.get())]['values']:
            if j == self.combobox_containers[int(self.v.get())].get():
                self.Algorithm_info["current_index"] = count
                break
            count += 1
        self.destroy()# 销毁窗口

    def cancel(self):
        self.Algorithm_info = None
        self.destroy()