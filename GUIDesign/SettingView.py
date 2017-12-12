from tkinter import *
from tkinter import ttk
import tkinter as tk
from tkinter.messagebox import *


class FlashRateFrame(Frame):
    def __init__(self, master, initial_para=None):
        Frame.__init__(self, master)
        self.root = master
        self.MODES = [
            ("Flash Rate On Choose", "0"),
            ("Flash Rate On Specific", "1"),
        ]
        self.v = StringVar()
        self.v.set("0")  # initialize
        self.choose_containers = [None, None]
        self.specific_containers = [None, None, None]
        self.choose_args = StringVar()
        self.entry_args = StringVar()
        self.flash_rate_dict = None
        self.choose_flash_rate = [100, 500, 1000, 2000, 5000]
        self.create_page()
        self.initial_flash_rate_dict(initial_para)

    def initial_flash_rate_dict(self, d):
        self.flash_rate_dict = {}
        if not d:
            self.flash_rate_dict["rate"] = 1000
            self.flash_rate_dict["mode"] = 0
            self.flash_rate_dict["current_index"] = 2
        else:
            for i in d:
                self.flash_rate_dict[i] = d[i]
            '''
            self.flash_rate_dict["rate"] = d["rate"]
            self.flash_rate_dict["mode"] = d["mode"]
            self.flash_rate_dict["current_index"] = d["current_index"]
            '''
        if self.flash_rate_dict["mode"] == 0:
            self.specific_containers[1]['state'] = DISABLED
            self.choose_containers[1]['state'] = NORMAL
            if self.flash_rate_dict["current_index"] in range(5):
                self.choose_containers[1].current(self.flash_rate_dict["current_index"])
        elif self.flash_rate_dict["mode"] == 1:
            self.specific_containers[1]['state'] = NORMAL
            self.choose_containers[1]['state'] = DISABLED
            self.entry_args.set(str(self.flash_rate_dict["rate"]))

    def create_page(self):
        st_row = 1
        for text, mode in self.MODES:
            b = Radiobutton(self, text=text,
                            variable=self.v, value=mode, command=self.radio_button_event)
            b.grid(row=st_row, column=1, pady=10, columnspan=4)
            st_row += 3
        self.choose_containers[0] = Label(self, text="FlashRate:", width=15)
        self.choose_containers[0].grid(row=2, column=1)
        self.choose_containers[1] = ttk.Combobox(self, textvariable=self.choose_args, state='readonly', width=10)
        self.choose_containers[1]['values'] = ('fast', 'less fast', 'normal', 'less low', 'low')
        self.choose_containers[1].grid(row=2, column=2)
        self.choose_containers[1].current(2)

        self.specific_containers[0] = Label(self, text="FlashRate:", width=15)
        self.specific_containers[0].grid(row=5, column=1)
        self.specific_containers[1] = Entry(self, textvariable=self.entry_args, width=15)
        self.specific_containers[1].grid(row=5, column=2)
        self.specific_containers[2] = Label(self, text="ms", width=5)
        self.specific_containers[2].grid(row=5, column=3)

        tk.Button(self, text="取消", command=self.cancel).grid(row=7, column=1, pady=10)
        tk.Button(self, text="确定", command=self.ok).grid(row=7, column=2, pady=10)

    def ok(self):
        self.flash_rate_dict = {}
        m = int(self.v.get())
        if m == 0:
            self.flash_rate_dict['mode'] = 0
            s = self.choose_args.get()
            count = 0
            for i in self.choose_containers[1]['values']:
                if s == i:
                    self.flash_rate_dict['rate'] = self.choose_flash_rate[count]
                    self.flash_rate_dict['current_index'] = count
                    break
                count += 1
        else:
            self.flash_rate_dict['mode'] = 1
            self.flash_rate_dict['rate'] = int(self.entry_args.get())
            self.flash_rate_dict['current_index'] = -1
        #showinfo(title='message', message='setting success')

    def cancel(self):
        self.flash_rate_dict = None

    def radio_button_event(self):
        m = int(self.v.get())
        if m == 0:
            self.specific_containers[1]['state'] = DISABLED
            self.choose_containers[1]['state'] = NORMAL
        elif m == 1:
            self.specific_containers[1]['state'] = NORMAL
            self.choose_containers[1]['state'] = DISABLED


class ClusteringFrame(Frame):
    def __init__(self, master, initial_para=None):
        Frame.__init__(self, master)
        self.root = master
        self.MODES = [
            ("Max iteration On Choose", "0"),
            ("Max iteration On Specific", "1"),
        ]
        self.v = StringVar()
        self.v.set("0")  # initialize
        self.cluster_k_args = StringVar()
        self.choose_args = StringVar()
        self.entry_args = StringVar()
        self.priors_args = [StringVar(), StringVar()]
        self.choose_containers = [None, None]
        self.specific_containers = [None, None]
        self.priors_container = None
        self.cluster_info_dict = None
        self.create_page()
        self.initial_cluster_info_dict(initial_para)

    def initial_cluster_info_dict(self, d):
        self.cluster_info_dict = {}
        if not d:
            self.cluster_info_dict["max_iteration"] = 100
            self.cluster_info_dict["iteration_mode"] = 0
            self.cluster_info_dict["iteration_current_index"] = 2
            self.cluster_info_dict["k"] = 0
            self.cluster_info_dict["priors_args1"] = 1
            self.cluster_info_dict["priors_args2_index"] = 0
            self.cluster_info_dict["priors"] = 0.1
        else:
            for i in d:
                self.cluster_info_dict[i] = d[i]
            '''
            self.cluster_info_dict["max_iteration"] = d["max_iteration"]
            self.cluster_info_dict["iteration_mode"] = d["iteration_mode"]
            self.cluster_info_dict["iteration_current_index"] = d["iteration_current_index"]
            self.cluster_info_dict["k"] = d["k"]
            self.cluster_info_dict["priors_args1"] = d["priors_args1"]
            self.cluster_info_dict["priors_args2_index"] = d["priors_args2_index"]
            self.cluster_info_dict["priors"] = d["priors"]
            '''

        self.cluster_k_args.set(str(self.cluster_info_dict["k"]))
        self.priors_args[0].set(str(self.cluster_info_dict["priors_args1"]))
        self.priors_container.current(self.cluster_info_dict["priors_args2_index"])
        if self.cluster_info_dict["iteration_mode"] == 0:
            self.specific_containers[1]['state'] = DISABLED
            self.choose_containers[1]['state'] = NORMAL
            if self.cluster_info_dict["iteration_current_index"] in range(5):
                self.choose_containers[1].current(self.cluster_info_dict["iteration_current_index"])
        elif self.cluster_info_dict["iteration_mode"] == 1:
            self.specific_containers[1]['state'] = NORMAL
            self.choose_containers[1]['state'] = DISABLED
            self.entry_args.set(str(self.cluster_info_dict["max_iteration"]))

    def create_page(self):
        Label(self, text="number of clusters:", width=20).grid(row=1, column=1)
        Entry(self, textvariable=self.cluster_k_args, width=5).grid(row=1, column=2)

        st_col = 1
        for text, mode in self.MODES:
            b = Radiobutton(self, text=text,
                            variable=self.v, value=mode, command=self.radio_button_event)
            b.grid(row=2, column=st_col, pady=10, columnspan=3)
            st_col += 3

        self.choose_containers[0] = Label(self, text="Max Iteration:", width=10)
        self.choose_containers[0].grid(row=3, column=1)
        self.choose_containers[1] = ttk.Combobox(self, textvariable=self.choose_args, state='readonly', width=5)
        self.choose_containers[1]['values'] = ('10', '50', '100', '200', '300', '500')
        self.choose_containers[1].grid(row=3, column=2)
        self.choose_containers[1].current(2)

        self.specific_containers[0] = Label(self, text="Max Iteration:", width=10)
        self.specific_containers[0].grid(row=3, column=4)
        self.specific_containers[1] = Entry(self, textvariable=self.entry_args, width=5)
        self.specific_containers[1].grid(row=3, column=5)

        Label(self, text="Prior Value:", width=15).grid(row=4, column=1)
        Entry(self, textvariable=self.priors_args[0], width=5).grid(row=4, column=2)
        Label(self, text="x", width=1).grid(row=4, column=3)
        self.priors_container = ttk.Combobox(self, textvariable=self.priors_args[1], state='readonly', width=10)
        self.priors_container['values'] = ('e^-1', 'e^-2', 'e^-3', 'e^-4', 'e^-5', 'e^-6')
        self.priors_container.grid(row=4, column=4)
        self.priors_container.current(0)

        tk.Button(self, text="取消", command=self.cancel).grid(row=7, column=1, pady=10)
        tk.Button(self, text="确定", command=self.ok).grid(row=7, column=2, pady=10)

        return

    def radio_button_event(self):
        m = int(self.v.get())
        if m == 0:
            self.specific_containers[1]['state'] = DISABLED
            self.choose_containers[1]['state'] = NORMAL
        elif m == 1:
            self.specific_containers[1]['state'] = NORMAL
            self.choose_containers[1]['state'] = DISABLED
        return

    def ok(self):
        self.cluster_info_dict = {}
        self.cluster_info_dict["k"] = int(self.cluster_k_args.get())
        self.cluster_info_dict["priors_args1"] = self.priors_args[0].get()
        temp = self.priors_args[1].get()
        temp_c = 0
        for i in self.priors_container['values']:
            if i == self.priors_args[1].get():
                self.cluster_info_dict["priors_args2_index"] = temp_c
                break
            temp_c += 1
        self.cluster_info_dict["priors"] = float(self.priors_args[0].get()) / pow(10, temp_c+1)
        m = int(self.v.get())
        if m == 0:
            self.cluster_info_dict['iteration_mode'] = 0
            s = self.choose_args.get()
            self.cluster_info_dict['max_iteration'] = s
            count = 0
            for i in self.choose_containers[1]['values']:
                if s == i:
                    self.cluster_info_dict['iteration_current_index'] = count
                    break
                count += 1
        else:
            self.cluster_info_dict['iteration_mode'] = 1
            self.cluster_info_dict['max_iteration'] = int(self.entry_args.get())
            self.cluster_info_dict['iteration_current_index'] = -1
        #showinfo(title='message', message='setting success')

    def cancel(self):
        self.cluster_info_dict = None
