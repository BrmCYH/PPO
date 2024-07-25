'''
    Manipulat 
        keyboard
        snapshot
        
'''
import cv2
import mss
import numpy as np
import base64

import keyboard
class Interface:
    def __enter__(self):
        self._hook = keyboard(self._on_key_event)
    pass
    def __exit__(self,):
        keyboard.un
import time

def take_screenshot():
    with mss.mss() as sct:
        # 获取整个屏幕的像素数据
        while True:
            monitor = sct.monitors[0]  # 选择第一个显示器
            screenshot = sct.grab(monitor)
            screenshot_np = np.array(screenshot)
            yield screenshot_np
            time.sleep(1/30)