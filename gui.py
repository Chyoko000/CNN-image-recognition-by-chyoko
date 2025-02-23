#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wx
from test import evaluate_one_image
from PIL import Image
import numpy as np
import os

class HelloFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(HelloFrame, self).__init__(*args, **kw)

        self.SetBackgroundColour("#F0F0F0")  # 整个窗口背景颜色

        # 主面板
        panel = wx.Panel(self)
        panel.SetBackgroundColour("#FFFFFF")

        # 创建主垂直布局管理器
        vbox = wx.BoxSizer(wx.VERTICAL)

        # --- 顶部区域：标题与识别结果 ---
        topSizer = wx.BoxSizer(wx.HORIZONTAL)
        title = wx.StaticText(panel, label="花朵识别", style=wx.ALIGN_CENTER)
        font = title.GetFont()
        font.PointSize += 10
        font = font.Bold()
        title.SetFont(font)
        title.SetForegroundColour("#333333")
        topSizer.Add(title, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=10)

        topSizer.AddStretchSpacer()

        self.result_text = wx.StaticText(panel, label="", style=wx.ALIGN_CENTER)
        result_font = self.result_text.GetFont()
        result_font.PointSize += 8
        self.result_text.SetFont(result_font)
        self.result_text.SetForegroundColour("#006400")
        topSizer.Add(self.result_text, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=10)

        vbox.Add(topSizer, flag=wx.EXPAND)

        # --- 按钮区域 ---
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSelect = wx.Button(panel, label="选择图片")
        btnSelect.SetBackgroundColour("#87CEFA")
        btnSelect.SetForegroundColour("#FFFFFF")
        btnSelect.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        btnSelect.Bind(wx.EVT_BUTTON, self.OnSelect)
        btnSizer.Add(btnSelect, flag=wx.ALL, border=10)
        vbox.Add(btnSizer, flag=wx.CENTER)

        # --- 图片展示区域 ---
        # 初始化时使用空位图占位，尺寸设置为 600x400
        self.image_ctrl = wx.StaticBitmap(panel, bitmap=wx.Bitmap(600, 400))
        vbox.Add(self.image_ctrl, flag=wx.ALL | wx.ALIGN_CENTER, border=10)

        panel.SetSizer(vbox)

        # 状态栏和菜单
        self.CreateStatusBar()
        self.SetStatusText("Welcome to Flower World")
        self.makeMenuBar()
        self.SetMinSize((800, 600))

    def makeMenuBar(self):
        fileMenu = wx.Menu()
        helloItem = fileMenu.Append(-1, "&Hello...\tCtrl-H", "显示帮助信息")
        fileMenu.AppendSeparator()
        exitItem = fileMenu.Append(wx.ID_EXIT)
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)
        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "Help")
        self.SetMenuBar(menuBar)
        self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnExit, exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)

    def OnExit(self, event):
        self.Close(True)

    def OnHello(self, event):
        wx.MessageBox("Hello again from wxPython", "Info", wx.OK | wx.ICON_INFORMATION)

    def OnAbout(self, event):
        wx.MessageBox("这是一个花朵识别示例程序，使用 wxPython 实现界面，\n调用 TensorFlow 模型进行花朵类别识别。",
                      "关于",
                      wx.OK | wx.ICON_INFORMATION)

    def OnSelect(self, event):
        # 设置文件选择对话框过滤器
        wildcard = "JPEG (*.jpg)|*.jpg|PNG (*.png)|*.png|All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "选择图片文件", os.getcwd(), "", wildcard, wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            file_path = dialog.GetPath()
            print("选择的图片路径：", file_path)
            self.initimage(file_path)
            # 打开图片并调整大小为模型要求的 64x64
            img = Image.open(file_path)
            img_resized = img.resize((64, 64))
            image_array = np.array(img_resized)
            # 调用评估函数获取预测结果
            result = evaluate_one_image(image_array)
            self.result_text.SetLabel(result)
            self.SetStatusText("识别完成")
        dialog.Destroy()

    def initimage(self, path):
        """
        在面板上显示选择的图片
        """
        imageShow = wx.Image(path, wx.BITMAP_TYPE_ANY)
        # 缩放图片至 600x400，并转换为位图显示
        bmp = imageShow.Scale(600, 400, wx.IMAGE_QUALITY_HIGH).ConvertToBitmap()
        self.image_ctrl.SetBitmap(bmp)
        self.Layout()

if __name__ == '__main__':
    app = wx.App(False)
    frm = HelloFrame(None, title='Flower World', size=(1000, 600))
    frm.Show()
    app.MainLoop()

