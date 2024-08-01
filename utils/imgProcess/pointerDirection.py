import cv2
import numpy as np
from tkinter import Tk, filedialog
from .dirClassifyModel.inference import getPointerDir


def getPointer(image, height, width):
    # 计算图像的中心点
    center_x, center_y = width / 2, height / 2
    # 提取中心20个像素的新图像
    # 假设提取的是一个40x40像素的区域
    half_size = 20
    start_x = int(max(center_x - half_size, 0))
    start_y = int(max(center_y - half_size, 0))
    end_x = int(min(center_x + half_size, width))
    end_y = int(min(center_y + half_size, height))

    center_image = image[start_y:end_y, start_x:end_x]
    # cv2.imshow("pic",center_image)
    # cv2.waitKey(25)
    return center_image

def extractPointer(image):
    # HSV方法
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)

    # 定义蓝色的HSV范围
    lower_blue = np.array([70, 100, 200])
    upper_blue = np.array([130, 255, 255])

    # 创建蓝色掩码
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 检测轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 假设最大的轮廓是三角形区域
    contour = max(contours, key=cv2.contourArea)


    # 设置轮廓近似精度，epsilon 值越小，轮廓越精确
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    output_image = image

    # 绘制近似轮廓（红色）
    cv2.drawContours(output_image, [approx], -1, (0, 0, 255), 1)

    return output_image

    # RGB 方法
    # 分离B、G、R通道
    # B, G, R = cv2.split(image)
    # # 0, 217, 245
    # # 30, 197, 233
    # # 5, 232, 254
    # # 蓝色通道的阈值过滤
    # _, blue_mask = cv2.threshold(B, 233, 255, cv2.THRESH_BINARY)
    # cv2.imshow("blue_mask", blue_mask)
    #
    # # 绿色通道的反向阈值过滤
    # _, green_mask = cv2.threshold(G, 255, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("green_mask", green_mask)
    #
    # # 红色通道的反向阈值过滤
    # _, red_mask = cv2.threshold(R, 255, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("red_mask", red_mask)
    #
    # # 结合所有通道的掩码
    # combined_mask = cv2.bitwise_and(blue_mask, green_mask)
    # combined_mask = cv2.bitwise_and(combined_mask, red_mask)
    #
    # # 应用形态学操作去除噪声
    # # kernel = np.ones((5, 5), np.uint8)
    # # mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    # # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("mask", combined_mask)
    # cv2.waitKey(0)
    # return combined_mask

def select_file(title="Select file"):
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    return file_path

def getPointerDirection(image):
    # 调整图片大小
    image = cv2.resize(image, (218, 218))
    # 提取箭头区域
    roi = getPointer(image, image.shape[0], image.shape[1])
    # 获取箭头方向
    result = getPointerDir(roi)
    return result

def main():
    image_path = select_file()

    result = getPointerDirection(image_path)

    # 显示结果
    image = cv2.imread(image_path)
    cv2.putText(image, str(result), (50, 50), color=(255, 255, 255), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN,
                thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow("image", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()



