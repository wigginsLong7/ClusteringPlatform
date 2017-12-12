from GUIDesign.SettingView import *
import tkinter as tk
from tkinter import *


class SettingPage(tk.Toplevel):
    def __init__(self, master, initial_para=None):
        super().__init__()
        self.title('其他设置')
        self.root = master
        self.geometry('%dx%d' % (450, 400))
        self.SettingInfo = None
        self.initial_setting_info(initial_para)
        self.ClusteringPage = ClusteringFrame(self, self.SettingInfo["ClusterInfo"])  # 创建不同Frame
        self.FlashRatePage = FlashRateFrame(self, self.SettingInfo["FlashRate"])
        self.create_page()
        self.protocol("WM_DELETE_WINDOW", self.quit_setting_info)

    def initial_setting_info(self, d):
        self.SettingInfo = {}
        if not d:
            self.SettingInfo["FlashRate"] = {}
            self.SettingInfo["FlashRate"]["rate"] = 1000
            self.SettingInfo["FlashRate"]["mode"] = 0
            self.SettingInfo["FlashRate"]["current_index"] = 2
            self.SettingInfo["ClusterInfo"] = {}
            self.SettingInfo["ClusterInfo"]["max_iteration"] = 100
            self.SettingInfo["ClusterInfo"]["iteration_mode"] = 0
            self.SettingInfo["ClusterInfo"]["iteration_current_index"] = 2
            self.SettingInfo["ClusterInfo"]["k"] = 0
            self.SettingInfo["ClusterInfo"]["priors_args1"] = 1
            self.SettingInfo["ClusterInfo"]["priors_args2_index"] = 0
            self.SettingInfo["ClusterInfo"]["priors"] = 0.1
        else:
            self.SettingInfo["FlashRate"] = {}
            self.SettingInfo["ClusterInfo"] = {}
            for i in d['FlashRate']:
                self.SettingInfo["FlashRate"][i] = d['FlashRate'][i]
            for j in d['ClusterInfo']:
                self.SettingInfo["ClusterInfo"][j] = d['ClusterInfo'][j]
        return

    def create_page(self):

        self.ClusteringPage.grid(row=1, column=1)  # 默认显示数据录入界面
        menubar = Menu(self)
        menubar.add_command(label='Clustering', command=self.clustering_setting)
        menubar.add_command(label='Color')
        menubar.add_command(label='FlashRate', command=self.flash_rate_setting)
        menubar.add_command(label='Other')
        menubar.add_command(label='About')
        menubar.add_command(label='Exit', command=self.quit_setting_info)

        self['menu'] = menubar  # 设置菜单栏
        return

    def quit_setting_info(self):
        for i in self.SettingInfo["FlashRate"]:
            self.SettingInfo["FlashRate"][i] = self.FlashRatePage.flash_rate_dict[i]
        for i in self.SettingInfo["ClusterInfo"]:
            self.SettingInfo["ClusterInfo"][i] = self.ClusteringPage.cluster_info_dict[i]
        self.destroy()

    def clustering_setting(self):
        self.ClusteringPage = ClusteringFrame(self, self.SettingInfo["ClusterInfo"])  # 创建不同Frame
        self.ClusteringPage.grid(row=1, column=1)
        '''
        self.SettingInfo["FlashRate"]["rate"] = self.FlashRatePage.flash_rate_dict["rate"]
        self.SettingInfo["FlashRate"]["mode"] = self.FlashRatePage.flash_rate_dict["mode"]
        self.SettingInfo["FlashRate"]["current_index"] = self.FlashRatePage.flash_rate_dict["current_index"]
        '''

        for i in self.SettingInfo["FlashRate"]:
            self.SettingInfo["FlashRate"][i] = self.FlashRatePage.flash_rate_dict[i]
        # self.PersonPage.pack_forget()
        # self.CombatPage.pack_forget()

        self.FlashRatePage.grid_forget()
        # self.MallPage.pack_forget()
        # self.SettingPage.pack_forget()
        # self.AboutPage.pack_forget()

    def flash_rate_setting(self):
        self.FlashRatePage = FlashRateFrame(self, self.SettingInfo["FlashRate"])
        self.FlashRatePage.grid(row=1, column=1)
        '''
        self.SettingInfo["ClusterInfo"]["max_iteration"] = self.ClusteringPage.cluster_info_dict["max_iteration"]
        self.SettingInfo["ClusterInfo"]["iteration_mode"] = self.ClusteringPage.cluster_info_dict["iteration_mode"]
        self.SettingInfo["ClusterInfo"]["iteration_current_index"] = self.ClusteringPage.cluster_info_dict["iteration_current_index"]
        self.SettingInfo["ClusterInfo"]["k"] = self.ClusteringPage.cluster_info_dict["k"]
        self.SettingInfo["ClusterInfo"]["priors_args1"] = self.ClusteringPage.cluster_info_dict["priors_args1"]
        self.SettingInfo["ClusterInfo"]["priors_args2_index"] = self.ClusteringPage.cluster_info_dict["priors_args2_index"]
        self.SettingInfo["ClusterInfo"]["priors"] = self.ClusteringPage.cluster_info_dict["priors"]
        '''
        for i in self.SettingInfo["ClusterInfo"]:
            self.SettingInfo["ClusterInfo"][i] = self.ClusteringPage.cluster_info_dict[i]
        self.ClusteringPage.grid_forget()

        # self.PersonPage.pack_forget()
        # self.CombatPage.pack_forget()
        # self.MallPage.pack_forget()
        # self.SettingPage.pack_forget()
        # self.AboutPage.pack_forget()