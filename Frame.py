# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Frame.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(740, 511)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 30, 211, 411))
        self.groupBox.setObjectName("groupBox")
        self.GetGestureButton = QtWidgets.QPushButton(self.groupBox)
        self.GetGestureButton.setGeometry(QtCore.QRect(20, 40, 171, 51))
        self.GetGestureButton.setObjectName("GetGestureButton")
        self.HelpButton = QtWidgets.QPushButton(self.groupBox)
        self.HelpButton.setGeometry(QtCore.QRect(20, 310, 171, 51))
        self.HelpButton.setObjectName("HelpButton")
        self.ExcuteGestureButton = QtWidgets.QPushButton(self.groupBox)
        self.ExcuteGestureButton.setGeometry(QtCore.QRect(20, 220, 171, 51))
        self.ExcuteGestureButton.setObjectName("ExcuteGestureButton")
        self.JudgeButton = QtWidgets.QPushButton(self.groupBox)
        self.JudgeButton.setGeometry(QtCore.QRect(20, 130, 171, 51))
        self.JudgeButton.setObjectName("JudgeButton")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(300, 30, 371, 291))
        self.groupBox_2.setObjectName("groupBox_2")
        self.LitResultlabel = QtWidgets.QLabel(self.groupBox_2)
        self.LitResultlabel.setGeometry(QtCore.QRect(60, 50, 241, 21))
        self.LitResultlabel.setText("")
        self.LitResultlabel.setObjectName("LitResultlabel")
        self.ImaResultlabel = QtWidgets.QLabel(self.groupBox_2)
        self.ImaResultlabel.setGeometry(QtCore.QRect(70, 80, 221, 171))
        self.ImaResultlabel.setText("")
        self.ImaResultlabel.setObjectName("ImaResultlabel")
        self.CloseButton = QtWidgets.QPushButton(self.centralwidget)
        self.CloseButton.setGeometry(QtCore.QRect(540, 370, 111, 31))
        self.CloseButton.setObjectName("CloseButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 740, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.CloseButton.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "操作按钮"))
        self.GetGestureButton.setText(_translate("MainWindow", "获取手势"))
        self.HelpButton.setText(_translate("MainWindow", "操作提示"))
        self.ExcuteGestureButton.setText(_translate("MainWindow", "执行手势"))
        self.JudgeButton.setText(_translate("MainWindow", "判断手势"))
        self.groupBox_2.setTitle(_translate("MainWindow", "结果显示"))
        self.CloseButton.setText(_translate("MainWindow", "关闭"))

