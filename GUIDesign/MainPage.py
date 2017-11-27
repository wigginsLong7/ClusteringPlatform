from tkinter import *
from tkinter.messagebox import *
from PIL import ImageTk, Image
from GUIDesign.Storekernel import StoreKernel
from OSHandler import OSHandler
import OSEnum
import GUIDesign.GUIEnum as GUIEnum
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import matplotlib.pyplot as plt
from algorithms_kernel.ClusteringHandler import *
from GUIDesign.AlgorithmPage import AlgorithmPage
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk


class MainPage(object):
    def __init__(self, master=None):
        self.root = master
        self.root.geometry('%dx%d' % (1000, 800))

        self.process_string = ""
        self.result_string = ""
        self.data_path = ""
        self.Entry_page_index_box = StringVar()
        self.Run_State_String = StringVar()
        self.process_roll_text = None
        self.result_roll_text = None
        self.process_canvas = None
        self.result_canvas = None
        self.matlab_process_canvas = None
        self.matlab_result_canvas = None
        self.canvas_page_index = 0
        self.run_flag = False
        self.process_fig_map = None
        self.result_fig_map = None
        self.show_process_bar = None
        self.map_set = []
        self.Entry_box_process_content = ""
        self.Entry_box_result_content = ""

        self.kernel_content = StoreKernel()
        self.ico_package = []
        self.page = Frame(self.root)
        #self.page.pack()
        self.create_page()
        self.timer_event()
        self.Run_State_String.set("State: Ready")

    def load_ico(self):
        self.ico_package.append(ImageTk.PhotoImage(Image.open('.\Image_Ico\Image_exit.gif')))
        self.ico_package.append(ImageTk.PhotoImage(Image.open('.\Image_Ico\Image_open.gif')))
        self.ico_package.append(ImageTk.PhotoImage(Image.open('.\Image_Ico\Image_save.gif')))
        self.ico_package.append(ImageTk.PhotoImage(Image.open('.\Image_Ico\Image_run.gif')))
        self.ico_package.append(ImageTk.PhotoImage(Image.open('.\Image_Ico\Image_pause.gif')))
        self.ico_package.append(ImageTk.PhotoImage(Image.open('.\Image_Ico\Image_clear.gif')))
        self.ico_package.append(ImageTk.PhotoImage(Image.open('.\Image_Ico\Image_prev.gif')))
        self.ico_package.append(ImageTk.PhotoImage(Image.open('.\Image_Ico\Image_back.gif')))
        self.ico_package.append(ImageTk.PhotoImage(Image.open('.\Image_Ico\Image_setting.gif')))
        self.ico_package.append(ImageTk.PhotoImage(Image.open('.\Image_Ico\Image_algorithm.gif')))
        self.ico_package.append(ImageTk.PhotoImage(Image.open('.\Image_Ico\Image_continue.gif')))

    def hello(self):
        print("Hello")

    def go_to_prev_frame(self):
        if self.canvas_page_index > 0:
            self.canvas_page_index -= 1
            self.print_image(self.canvas_page_index, GUIEnum.PRINT_DATA_CENTER, GUIEnum.PROCESS_CANVAS)
            self.update_scrolled_text(self.canvas_page_index, GUIEnum.PRINT_DATA_CENTER, GUIEnum.PROCESS_CANVAS)
            self.Entry_page_index_box.set(str(self.canvas_page_index))
        else:
            pass

    def go_to_back_frame(self):
        if self.canvas_page_index < len(self.kernel_content.center_set)-1:
            self.canvas_page_index += 1
            self.print_image(self.canvas_page_index, GUIEnum.PRINT_DATA_CENTER, GUIEnum.PROCESS_CANVAS)
            self.update_scrolled_text(self.canvas_page_index, GUIEnum.PRINT_DATA_CENTER, GUIEnum.PROCESS_CANVAS)
            self.Entry_page_index_box.set(str(self.canvas_page_index))
        else:
            pass

    def quit(self):
        exit()

    def load_file(self, path):
        if path.find('txt') != -1:
            self.kernel_content.data = OSHandler.load_data(path, OSEnum.TXT_FORMAT)
        elif path.find('mat') != -1:
            self.kernel_content.data = OSHandler.load_data(path, OSEnum.MAT_FORMAT)
        else:
            return False
        return True

    def timer_event(self):
        if self.run_flag and len(self.kernel_content.center_set):
            if self.canvas_page_index >= len(self.kernel_content.center_set):
                self.canvas_page_index = len(self.kernel_content.center_set) - 1
            self.print_image(self.canvas_page_index, GUIEnum.PRINT_DATA_CENTER, GUIEnum.PROCESS_CANVAS)
            self.update_scrolled_text(self.canvas_page_index, GUIEnum.PRINT_DATA_CENTER, GUIEnum.PROCESS_CANVAS)
            self.canvas_page_index += 1
            self.Entry_page_index_box.set(str(self.canvas_page_index))
        self.hello()
        self.root.after(1000, self.timer_event)

    def open_file_dialog(self):
        fname = askopenfilename(filetypes=(("Txt files", "*.txt"), ("Mat files", "*.mat"), ("All files", "*.*")))
        if fname == "":
            return
        self.data_path = fname
        self.load_file(fname)
        self.print_image()
        self.update_scrolled_text()

    def update_scrolled_text(self, center_index=-1, mode=GUIEnum.PRINT_ONLY_DATA, canvas_index=GUIEnum.RESULT_CANVAS):
        d = np.squeeze(np.asarray(self.kernel_content.data))
        data_dimension = len(d[0, :])
        data_num = len(d[:, 0])
        show_dimension = min(GUIEnum.MAX_DATA_DIMENSION_SHOW_NUMBER, data_dimension)
        self.result_string = ""
        self.result_string = "Data_path: " + self.data_path + "\n" + "Data_num: " + str(
            data_num) + "\n" + "Data_dimension: " + str(data_dimension) + "\n"
        c_index = center_index
        mu = None
        if not len(self.kernel_content.center_set):
            pass
        else:
            if center_index >= len(self.kernel_content.center_set):
                c_index = len(self.kernel_content.center_set) - 1
        if c_index > -1:
            mu = np.squeeze(np.asarray(self.kernel_content.center_set[c_index]))
        # pca 降维
        # change d and mu

        if mode == GUIEnum.PRINT_ONLY_DATA:
            if canvas_index != GUIEnum.RESULT_CANVAS:
                return False
            self.result_roll_text.delete(0.0, END)
            self.result_roll_text.insert(END, self.result_string)
            # 源数据的资料
        else:
            center_k = len(mu[:, 0])
            self.result_string += ("Center_num: " + str(center_k) + "\n")
            for i in range(center_k):
                self.result_string += "Center " + str(i) + " ("
                for j in range(show_dimension):
                    self.result_string += (str(round(mu[i][j], 2)) + ",")
                if data_dimension > show_dimension:
                    self.result_string += "...)"
                else:
                    self.result_string += ")"
                self.result_string += "， Prior: " + str(round(self.kernel_content.prior_set[c_index][0, i], 3)) + "\n"
            self.result_string += "Likelihood: "
            if canvas_index == GUIEnum.RESULT_CANVAS:
                self.result_roll_text.delete(0.0, END)
                self.result_roll_text.insert(END, self.result_string)
            elif canvas_index == GUIEnum.PROCESS_CANVAS:
                self.process_roll_text.delete(0.0, END)
                self.process_roll_text.insert(END, self.result_string)
            else:
                return False
        return True

    def print_image(self, center_index=-1, mode=GUIEnum.PRINT_ONLY_DATA, canvas_index=GUIEnum.RESULT_CANVAS):
        d = np.squeeze(np.asarray(self.kernel_content.data))
        c_index = center_index
        if not len(self.kernel_content.center_set):
            pass
        else:
            if center_index >= len(self.kernel_content.center_set):
                c_index = len(self.kernel_content.center_set) - 1
        if c_index > -1:
            mu = np.squeeze(np.asarray(self.kernel_content.center_set[c_index]))
        plt.interactive(False)
        if canvas_index == GUIEnum.PROCESS_CANVAS:
            self.process_fig_map = plt.figure(figsize=(4, 4))
            self.map_set.append(self.process_fig_map)
        else:
            self.result_fig_map = plt.figure(figsize=(4, 4))
            self.map_set.append(self.result_fig_map)
        plt.scatter(d[:, 0], d[:, 1], s=30, c='red', marker='o', alpha=0.5, label='data point')
        if mode != GUIEnum.PRINT_ONLY_DATA:
            plt.scatter(mu[:, 0], mu[:, 1], s=50, c='blue', marker='o', alpha=0.5, label='center point')
        plt.title('data ')
        plt.xlabel('variables x')
        plt.ylabel('variables y')
        plt.legend(loc='upper right')  # 这个必须有，没有你试试看
        if canvas_index == GUIEnum.PROCESS_CANVAS:
            if self.matlab_process_canvas:
                self.matlab_process_canvas.get_tk_widget().destroy()
            self.matlab_process_canvas = FigureCanvasTkAgg(self.process_fig_map, master=self.process_canvas)
            #self.matlab_process_canvas.get_tk_widget().pack(expand=1)
            self.matlab_process_canvas.get_tk_widget().grid(row=3, column=0)
            self.matlab_process_canvas.draw()
        else:
            if self.matlab_result_canvas:
                self.matlab_result_canvas.get_tk_widget().destroy()
            self.matlab_result_canvas = FigureCanvasTkAgg(self.result_fig_map, master=self.result_canvas)
            #self.matlab_result_canvas.get_tk_widget().pack(expand=1)
            self.matlab_result_canvas.get_tk_widget().grid(row=3, column=1)
            self.matlab_result_canvas.draw()

    def thread_start(self):
        g_th = threading.Thread(target=self.run, args=())
        g_th.setDaemon(True)  # 守护线程
        g_th.start()
        return

    def run(self):
        self.run_flag = True
        self.Run_State_String.set("State: Run")
        args = ClusteringHandlerBOGMM.return_parameters_args()
        args['mode'] = AlgorithmsEnum.LAGRANGE_YING_YANG_ALTERNATION_MODE
        args['prior_threshold'] = 0.2
        args['args1'] = 2
        args['seed_num'] = 1
        sy, center, label, n_step, old_likelihood = ClusteringHandlerBOGMM.EM_Solution(self.kernel_content.data.T, 10, args,
                 self.kernel_content.center_set, self.kernel_content.data_id_set, self.kernel_content.prior_set, self.kernel_content.sigma_set)

        self.print_image(len(self.kernel_content.center_set)-1, GUIEnum.PRINT_DATA_CENTER, GUIEnum.RESULT_CANVAS)
        self.update_scrolled_text(len(self.kernel_content.center_set)-1, GUIEnum.PRINT_DATA_CENTER, GUIEnum.RESULT_CANVAS)
        self.run_flag = False
        self.Run_State_String = "State: finish"
        print(n_step)
        return

    def stop_processing(self):
        self.run_flag = False
        self.Run_State_String.set("State: Pause")

    def clear_all_graph(self):
        self.run_flag = False
        self.matlab_process_canvas.get_tk_widget().destroy()
        self.matlab_result_canvas.get_tk_widget().destroy()
        plt.clf()

    def pop_up_algorithm_frame(self):
        algorithm_dialog = AlgorithmPage()
        self.page.wait_window(algorithm_dialog)  # 这一句很重要！！！
        print(algorithm_dialog.user_info)

    '''
    def open_algoritm_settings(self):

    def open_option_settings(self):


    def test_pause(self):

    def test_continue(self):

    def frame_jump(self):

    def quit(self):

    def gaussian_build(self):
    '''
    def create_menu(self):
        menubar = Menu(self.root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="打开", command=self.open_file_dialog)
        filemenu.add_command(label="关闭", command=self.hello)
        filemenu.add_separator()
        filemenu.add_command(label="保存", command=self.hello)
        filemenu.add_command(label="另存为", command=self.hello)
        filemenu.add_separator()
        filemenu.add_command(label="退出", command=self.hello)
        menubar.add_cascade(label="文件", menu=filemenu)

        option_menu = Menu(menubar, tearoff=0)
        option_menu.add_command(label="算法设置", command=self.hello)
        option_menu.add_command(label="软件设置", command=self.hello)
        option_menu.add_command(label="窗口设置", command=self.hello)
        menubar.add_cascade(label="选项设置", menu=option_menu)

        play_menu = Menu(menubar, tearoff=0)
        play_menu.add_command(label="播放", command=self.hello)
        play_menu.add_command(label="关闭", command=self.hello)
        play_menu.add_separator()
        play_menu.add_command(label="下一帧", command=self.hello)
        play_menu.add_command(label="上一帧", command=self.hello)
        menubar.add_cascade(label="播放选项", menu=play_menu)

        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="关于", command=self.hello)
        helpmenu.add_separator()
        helpmenu.add_command(label="使用方法", command=self.hello)
        menubar.add_cascade(label="帮助", menu=helpmenu)
        helpmenu = Menu(menubar, tearoff=0)
        self.root.config(menu=menubar)
        return

    def create_toolbar(self):
        '''
        toolframe = Frame(self.root, height=20)
        frame = Frame(toolframe)
        Button(frame, width=20, image=self.ico_package[0], command=self.hello).grid(row=0, column=0, padx=1, pady=1,
                                                                              sticky=E)
        Button(frame, width=20, image=self.ico_package[1], command=self.hello).grid(row=0, column=1, padx=1, pady=1,
                                                                              sticky=E)
        Button(frame, width=20, image=self.ico_package[2], command=self.hello).grid(row=0, column=2, padx=1, pady=1,
                                                                              sticky=E)
        frame.pack(side=LEFT)
        toolframe.pack(fill=X)
        '''
        toolbar = Frame(self.root, height=25)
        shortButton = Button(toolbar, image=self.ico_package[0], command=self.quit, relief=FLAT)
        shortButton.grid(row=2, column=0, padx=5)
        #shortButton.pack(side=LEFT, padx=5, pady=5)

        shortButton = Button(toolbar, image=self.ico_package[1], command=self.open_file_dialog, relief=FLAT)
        #shortButton.pack(side=LEFT, padx=5, pady=5)
        shortButton.grid(row=2, column=1, padx=5)

        shortButton = Button(toolbar, image=self.ico_package[2], command=self.open_file_dialog, relief=FLAT) # save
        #shortButton.pack(side=LEFT, padx=5, pady=5)
        shortButton.grid(row=2, column=2, padx=5)

        shortButton = Button(toolbar, image=self.ico_package[3], command=self.thread_start, relief=FLAT)
        #shortButton.pack(side=LEFT, padx=5, pady=5)
        shortButton.grid(row=2, column=3, padx=5)

        shortButton = Button(toolbar, image=self.ico_package[4], command=self.stop_processing, relief=FLAT)
        #shortButton.pack(side=LEFT, padx=5, pady=5)
        shortButton.grid(row=2, column=4, padx=5)

        shortButton = Button(toolbar, image=self.ico_package[5], command=self.clear_all_graph, relief=FLAT)
        #shortButton.pack(side=LEFT, padx=5, pady=5)
        shortButton.grid(row=2, column=5, padx=5)

        shortButton = Button(toolbar, image=self.ico_package[6], command=self.go_to_prev_frame, relief=FLAT)
        #shortButton.pack(side=LEFT, padx=5, pady=5)
        shortButton.grid(row=2, column=6, padx=5)

        shortButton = Button(toolbar, image=self.ico_package[7], command=self.go_to_back_frame, relief=FLAT)
        #shortButton.pack(side=LEFT, padx=5, pady=5)
        shortButton.grid(row=2, column=7, padx=5)

        shortButton = Button(toolbar, image=self.ico_package[8], command=self.open_file_dialog, relief=FLAT)
        #shortButton.pack(side=LEFT, padx=5, pady=5)
        shortButton.grid(row=2, column=8, padx=5)

        shortButton = Button(toolbar, image=self.ico_package[9], command=self.pop_up_algorithm_frame, relief=FLAT)
        #shortButton.pack(side=LEFT, padx=5, pady=5)
        shortButton.grid(row=2, column=9, padx=5)

        Label(self.root, text="FrameIndex:", width=10).grid(row=2, column=10, padx=5)
        Entry(self.root, textvariable=self.Entry_page_index_box, width=10).grid(row=2, column=11, padx=5)
        Label(self.root, textvariable=self.Run_State_String, width=10).grid(row=2, column=12, padx=5)
        toolbar.grid(row=2)

        #toolbar.pack(expand=NO, fill=X)

    def create_page(self):
        self.load_ico()
        self.create_menu()
        self.create_toolbar()
        self.process_canvas = Canvas(self.root, width=400, height=400, bg='white')
        self.result_canvas = Canvas(self.root, width=400, height=400, bg='white')
        self.process_canvas.grid(row=3, column=0, rowspan=8, columnspan=8, pady=10, padx=15)
        self.result_canvas.grid(row=3, column=8, rowspan=8, columnspan=8, pady=10, padx=15)
        #self.process_canvas.pack(side=LEFT, padx=20, pady=10, anchor=NW)
        #self.result_canvas.pack(side=LEFT, padx=20, pady=10,  anchor=NE)
        self.process_roll_text = ScrolledText(self.root, width=40, height=10, background='#ffffff')
        self.process_roll_text.grid(row=11, column=0, columnspan=8, pady=10, padx=5)
        self.result_roll_text = ScrolledText(self.root, width=40, height=10, background='#ffffff')
        self.result_roll_text.grid(row=11, column=8, columnspan=8, pady=10, padx=5)


        self.show_process_bar = ttk.Progressbar(self.root, length=400, maximum=100).grid(row=13, column=0, columnspan=10,pady=10)
        Label(self.root, text="Process Rate", width=10).grid(row=13, column=10, columnspan=1, pady=5)

        #m.pack(side=LEFT, padx=20, pady=10, anchor=NW)

