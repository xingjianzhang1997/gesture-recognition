import cv2 as cv

img_roi_y = 30
img_roi_x = 200
img_roi_height = 350  # [2]设置ROI区域的高度
img_roi_width = 350  # [3]设置ROI区域的宽度
capture = cv.VideoCapture(0)
index = 1
num = 150
while True:
    ret, frame = capture.read()
    if ret is True:
        img_roi = frame[img_roi_y:(img_roi_y + img_roi_height), img_roi_x:(img_roi_x + img_roi_width)]
        cv.imshow("frame", img_roi)
        index += 1
        if index % 6 == 0:   # 每20帧保存一次图像
            num += 1
            cv.imwrite("D:/python/deep-learning/Gesture-recognition/data/test/"
                       + "gesture_5."+str(num) + ".jpg", img_roi)
        c = cv.waitKey(50)  # 每50ms判断一下键盘的触发。  0则为无限等待。
        if c == 27:  # 在ASCII码中27表示ESC键，ord函数可以将字符转换为ASCII码。
            break
        if index == 300:
            break
    else:
        break

cv.destroyAllWindows()
capture.release()
