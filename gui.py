#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wx
from test import evaluate_one_image  # 请确保该模块中实现了基于 TF2.x 的 evaluate_one_image 函数
from PIL import Image
import numpy as np
import os


class HelloFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(HelloFrame, self).__init__(*args, **kw)

        # 创建主面板
        self.pnl = wx.Panel(self)

        # 标题文本
        title = wx.StaticText(self.pnl, label="花朵识别", pos=(200, 10))
        font = title.GetFont()
        font.PointSize += 10
        font = font.Bold()
        title.SetFont(font)

        # 创建选择图片的按钮
        btn = wx.Button(self.pnl, label="选择图片", pos=(50, 60))
        btn.Bind(wx.EVT_BUTTON, self.OnSelect)

        # 用于显示识别结果的文本控件（初始为空，可后续更新）
        self.result_text = wx.StaticText(self.pnl, label="", pos=(320, 10))
        result_font = self.result_text.GetFont()
        result_font.PointSize += 8
        self.result_text.SetFont(result_font)

        # 用于显示图片的控件
        self.image_ctrl = None

        # 创建菜单
        self.makeMenuBar()

        self.CreateStatusBar()
        self.SetStatusText("Welcome to Flower World")

    def makeMenuBar(self):
        fileMenu = wx.Menu()
        helloItem = fileMenu.Append(-1, "&Hello...\tCtrl-H",
                                    "显示帮助信息")
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
        # 设置文件选择对话框的过滤器
        wildcard = "JPEG (*.jpg)|*.jpg|All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "选择图片文件", os.getcwd(), "", wildcard, wx.FD_OPEN)

        if dialog.ShowModal() == wx.ID_OK:
            file_path = dialog.GetPath()
            print("选择的图片路径：", file_path)
            # 打开图片并展示
            img = Image.open(file_path)
            # 如果需要，可以在界面上展示原始图片
            self.initimage(file_path)

            # 将图片调整为模型需要的大小（这里使用 64x64）
            # 注意：PIL 中建议使用元组作为尺寸参数
            img_resized = img.resize((64, 64))
            # 将图片转换为 numpy 数组
            image_array = np.array(img_resized)
            # result = evaluate_one_image(image_array)
            # if result is None:
            #     result = "预测失败：未找到检查点文件"
            # 调用评估函数，得到预测结果
            result = evaluate_one_image(image_array)
            # 更新界面上的识别结果
            self.result_text.SetLabel(result)
            self.SetStatusText("识别完成")
        dialog.Destroy()

    def initimage(self, path):
        """
        在面板上显示选择的图片
        """
        # 加载图片
        imageShow = wx.Image(path, wx.BITMAP_TYPE_ANY)
        # 如果之前已经有图片显示，则先删除旧的控件
        if self.image_ctrl:
            self.image_ctrl.Destroy()
        # 创建 StaticBitmap 控件显示图片，设置一个合适的尺寸
        self.image_ctrl = wx.StaticBitmap(self.pnl, bitmap=imageShow.ConvertToBitmap(), pos=(50, 100), size=(600, 400))
        self.Layout()


if __name__ == '__main__':
    app = wx.App(False)
    frm = HelloFrame(None, title='Flower World', size=(1000, 600))
    frm.Show()
    app.MainLoop()
