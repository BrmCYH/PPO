import math
import cv2
import numpy as np
from tkinter import Tk, filedialog

# 创建图像字典
template_dict = {
    "bfl": "./data/template/bfl.png",
    "jds": "./data/template/jds.png",
    "zjh": "./data/template/zjh.png",
    "p": "./data/template/pointer.png",
}

def select_file(title="Select file"):
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    return file_path

# 模板匹配函数
def template_match(image, template, threshold):
    # 转换为灰度图像
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 模板匹配
    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # 获取匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        # 绘制匹配结果
        top_left = max_loc
        h, w = template_gray.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        print(f"匹配成功，匹配置信度为: {max_val}")
        return True, top_left, bottom_right
    else:
        print(f"匹配失败，匹配置信度为: {max_val}")
        return False, 0, 0

def getDistance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def getTargetDirection(image, pos: str, threshold):
    if pos in template_dict.keys():
        template_path = template_dict[pos]
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    else:
        print(f"没有找到{pos}模板，请先创建{pos}模板")
        return

    # 调整图片大小
    image = cv2.resize(image, (218, 218))
    template = cv2.resize(template, (32, 32))

    # 进行模板匹配，返回target位置
    result = template_match(image, template, threshold)

    # 获取图像中心点
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    # 如果匹配成功，计算方向
    if result[0] == True:
        top_left = result[1]
        bottom_right = result[2]
        target_x = (top_left[0] + bottom_right[0])/2
        target_y = (top_left[1] + bottom_right[1])/2

        # 计算向量 (dx, dy)
        dx, dy = target_x - center_x, target_y - center_y

        # 计算角度 (atan2 返回的角度范围是 [-π, π])
        angle = math.atan2(dy, dx)

        # 将角度转换为从正北方向开始的顺时针角度，即旋转90度（π/2）
        angle = angle + math.pi / 2

        # 将角度转换为 [0, 2π] 的范围
        if angle < 0:
            angle += 2 * math.pi

        # 将角度映射到12个时钟方向，每个时钟方向对应30度（π/6）
        direction = int((angle + math.pi / 12) // (math.pi / 6)) % 12

        # 将角度转换为角度制
        # angle_degrees = math.degrees(angle)
        # print(f'角度：{angle_degrees}')

        # 计算距离
        distance = getDistance(center_x, center_y, target_x, target_y)
        return direction, distance


def main():
    pos = input("输入希望匹配的地点：")
    if pos in template_dict.keys():
        image_path = select_file()
        targetDir = getTargetDirection(image_path, template_dict[pos], 0.5)
        print(f'方向：{targetDir}')
    else:
        print(f"没有找到{pos}模板，请先创建{pos}模板")


if __name__ == '__main__':
    main()