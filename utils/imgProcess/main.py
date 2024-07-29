import time

import cv2
from tkinter import Tk, filedialog

from imgBinarization import getBinaryImage
from pointerDirection import getPointerDirection
from templateMatching import getTargetDirection, template_dict


def select_file(title="Select file"):
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    return file_path

def getMap(image):
    img = image.copy()
    img = cv2.resize(img, (1602, 897))
    # 提取的是一个182x182像素的区域
    start_x = 49
    start_y = 8
    end_x = 231
    end_y = 190
    map = img[start_y:end_y, start_x:end_x]
    return map

def imgProcess(image):
    pos = 'bfl'

    start = time.time()
    # 截取小地图区域
    map = getMap(image)

    # 图像二值化处理
    img_binary = getBinaryImage(image, 5, 4)

    # 获取目标方向和距离
    target_dir, distance = getTargetDirection(map, pos, 0.5)

    # 获取箭头方向
    pointer_dir = getPointerDirection(map)

    end = time.time()
    print(f'耗时：{end - start}')
    print(f'箭头方向：{pointer_dir}')
    print(f'目标方向：{target_dir}')
    print(f'距离：{distance}')

    return pointer_dir, target_dir, distance, img_binary
