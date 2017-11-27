import tkinter as tk


class AlgorithmPage(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__()
        self.title('设置算法')
        self.root = master
        self.root.geometry('%dx%d' % (400, 300))
        self.name = None
        self.age = None
        self.user_info = None
        self.setup_ui()

    def setup_ui(self):
        row1 = tk.Frame(self)
        row1.pack(fill="x")
        tk.Label(row1, text='姓名：', width=8).pack(side=tk.LEFT)
        self.name = tk.StringVar()
        tk.Entry(row1, textvariable=self.name, width=20).pack(side=tk.LEFT)
        # 第二行
        row2 = tk.Frame(self)
        row2.pack(fill="x", ipadx=1, ipady=1)
        tk.Label(row2, text='年龄：', width=8).pack(side=tk.LEFT)
        self.age = tk.IntVar()
        tk.Entry(row2, textvariable=self.age, width=20).pack(side=tk.LEFT)
        # 第三行
        row3 = tk.Frame(self)
        row3.pack(fill="x")
        tk.Button(row3, text="取消", command=self.cancel).pack(side=tk.RIGHT)
        tk.Button(row3, text="确定", command=self.ok).pack(side=tk.RIGHT)

    def ok(self):
        self.user_info = [self.name.get(), self.age.get()] # 设置数据
        self.destroy()# 销毁窗口

    def cancel(self):
        self.user_info = None
        self.destroy()