import sys
import time
import serial  # 这个模块是通信模块
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QDateTime, QThread
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap, QPalette
from SaveGesture import *
from TestGesture import *
from Frame import *

global gesture_action

class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)   # 在界面文件Frame中以及根据界面自动定义了
        self.initUI()

    def initUI(self):
        # 给按钮连接槽函数（CloseButton在Frame中自动连接了）
        self.GetGestureButton.clicked.connect(self.GetGesture)
        self.JudgeButton.clicked.connect(self.JudgeGesture)
        self.ExcuteGestureButton.clicked.connect(self.ExcuteGesture)
        self.HelpButton.clicked.connect(self.Help)

        # 窗口设置美化
        self.setWindowTitle('手势识别')
        self.setWindowIcon(QIcon('./ges_ico/frame.ico'))
        self.resize(750, 485)

        # 线程操作用于显示时间
        self.initxianceng()

        # 单独给CloseButton添加标签
        self.CloseButton.setProperty('color', 'gray')  # 自定义标签
        self.GetGestureButton.setProperty('color', 'same')
        self.JudgeButton.setProperty('color', 'same')
        self.ExcuteGestureButton.setProperty('color', 'same')
        self.HelpButton.setProperty('color', 'same')

    # 定义槽函数
    def GetGesture(self):
        self.LitResultlabel.setText("")
        self.ImaResultlabel.setPixmap(QPixmap('./ges_ico/white.ico'))
        self.LitResultlabel.setAutoFillBackground(False)
        saveGesture()
        self.LitResultlabel.setText("已经将该图像保存在电脑本地")
        self.LitResultlabel.setAlignment(Qt.AlignCenter)

    def JudgeGesture(self):
        global gesture_action  # 要修改全局变量需要先在函数里面声明一下
        self.LitResultlabel.setText("正在调用卷积神经网络识别图像")
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        QApplication.processEvents()  # 这里需要刷新一下，否则上面的文字不显示
        gesture_num = evaluate_one_image()
        if gesture_num == 1:
            gesture_action = "1"
            self.result_show_1()
        elif gesture_num == 2:
            gesture_action = "2"
            self.result_show_2()
        elif gesture_num == 3:
            gesture_action = "3"
            self.result_show_3()
        elif gesture_num == 4:
            gesture_action = "4"
            self.result_show_4()
        elif gesture_num == 5:
            gesture_action = "5"
            self.result_show_5()

    def ExcuteGesture(self):
        self.serial_communicate()

    def Help(self):
        QMessageBox.information(self, "操作提示框", "获取手势：通过OpenCV和摄像头获取一张即时照片。\n"
                                "判断手势：通过之前训练好的参数和卷积神经网络判断手势。\n"
                                "执行手势：根据识别的手势姿态控制机械手作业。")

    def serial_communicate(self):
        """
        把gesture_action的字符（1~5）传给下位机
        """
        print(gesture_action)
        try:
            # G7电脑的左端口是COM4
            portx = "COM4"
            # 波特率，我STM32单片机设置的为115200
            bps = 115200
            # 超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
            timex = 5
            # 打开串口，并得到串口对象
            ser = serial.Serial(portx, bps, timeout=timex)
            # 写数据
            # result=ser.write("1".encode("utf-8"))   # encode("gbk")是中国中文的字符
            result = ser.write(gesture_action.encode("utf-8"))
            print("成功:", result)

            ser.close()  # 关闭串口

        except Exception as extioc:
            print("---异常---：", extioc)

    def result_show_1(self):
        self.LitResultlabel.setText("判断结果：该手势为剪刀")
        self.LitResultlabel.setAutoFillBackground(True)  # 允许上色
        palette = QPalette()                          # palette 调色板
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.LitResultlabel.setPalette(palette)
        self.ImaResultlabel.setToolTip('这是一个示意图片结果')  # 鼠标放在上面出现提示框
        self.ImaResultlabel.setPixmap(QPixmap('./ges_ico/ges1.ico'))
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        self.ImaResultlabel.setAlignment(Qt.AlignCenter)

    def result_show_2(self):
        self.LitResultlabel.setText("判断结果：该手势为石头")
        self.LitResultlabel.setAutoFillBackground(True)  # 允许上色
        palette = QPalette()                            # palette 调色板
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.LitResultlabel.setPalette(palette)
        self.ImaResultlabel.setToolTip('这是一个示意图片结果')  # 鼠标放在上面出现提示框
        self.ImaResultlabel.setPixmap(QPixmap('./ges_ico/ges3.ico'))
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        self.ImaResultlabel.setAlignment(Qt.AlignCenter)

    def result_show_3(self):
        self.LitResultlabel.setText("判断结果：该手势为布")
        self.LitResultlabel.setAutoFillBackground(True)
        palette = QPalette()                            # palette 调色板
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.LitResultlabel.setPalette(palette)
        self.ImaResultlabel.setToolTip('这是一个示意图片结果')  # 鼠标放在上面出现提示框
        self.ImaResultlabel.setPixmap(QPixmap('./ges_ico/ges2.ico'))
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        self.ImaResultlabel.setAlignment(Qt.AlignCenter)

    def result_show_4(self):
        self.LitResultlabel.setText("判断结果：该手势为OK")
        self.LitResultlabel.setAutoFillBackground(True)
        palette = QPalette()                            # palette 调色板
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.LitResultlabel.setPalette(palette)
        self.ImaResultlabel.setToolTip('这是一个示意图片结果')  # 鼠标放在上面出现提示框
        self.ImaResultlabel.setPixmap(QPixmap('./ges_ico/ges4.ico'))
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        self.ImaResultlabel.setAlignment(Qt.AlignCenter)

    def result_show_5(self):
        self.LitResultlabel.setText("判断结果：该手势为good")
        self.LitResultlabel.setAutoFillBackground(True)
        palette = QPalette()                            # palette 调色板
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.LitResultlabel.setPalette(palette)
        self.ImaResultlabel.setToolTip('这是一个示意图片结果')  # 鼠标放在上面出现提示框
        self.ImaResultlabel.setPixmap(QPixmap('./ges_ico/ges5.ico'))
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        self.ImaResultlabel.setAlignment(Qt.AlignCenter)

    def initxianceng(self):
        # 创建线程
        self.backend = BackendThread()
        # 信号连接槽函数
        self.backend.update_date.connect(self.handleDisplay)
        # 开始线程
        self.backend.start()

    # 将当期时间输出到文本框
    def handleDisplay(self, data):
        self.statusBar().showMessage(data)


# 后台线程更新时间
class BackendThread(QThread):
    update_date = pyqtSignal(str)

    def run(self):
        while True:
            date = QDateTime.currentDateTime()
            currTime = date.toString('yyyy-MM-dd hh:mm:ss')
            self.update_date.emit(str(currTime))
            time.sleep(1)  # 推迟执行的1秒


if __name__ == "__main__":
    app = QApplication(sys.argv)  # sys.argv是一个命令行参数列表
    myWin = MyMainWindow()
    myWin.setObjectName('Window')
    # 给窗口背景上色
    qssStyle = '''
              QPushButton[color='gray']{
              background-color:rgb(205,197,191)
              }
              QPushButton[color='same']{
              background-color:rgb(225,238,238)
              }
              #Window{
              background-color:rgb(162,181,205) 
              }
              '''
    myWin.setStyleSheet(qssStyle)
    myWin.show()
    sys.exit(app.exec_())

