import cv2


def saveGesture():
    """
    原始图像的大小为（480,640,3）
    保存的图像是切割后的图像（400,400,3）
    """
    cameraCapture = cv2.VideoCapture(0)
    success, frame = cameraCapture.read()
    if success is True:
        cv2.imwrite("./data/testImage/" + "test.jpg", frame)

    testImg = cv2.imread('./data/testImage/test.jpg')
    print(testImg.shape)  # G7笔记本的摄像头是（480，640,3） 高度，宽度，通道数

    img_roi_y = 30
    img_roi_x = 200
    img_roi_height = 350  # [2]设置ROI区域的高度
    img_roi_width = 350  # [3]设置ROI区域的宽度
    img_roi = testImg[img_roi_y:(img_roi_y + img_roi_height), img_roi_x:(img_roi_x + img_roi_width)]

    cv2.imshow("[ROI_Img]", img_roi)
    cv2.imwrite("./data/testImage/roi/" + "img_roi.jpg", img_roi)
    cv2.waitKey(0)
    cv2.destroyWindow("[ROI_Img]")
