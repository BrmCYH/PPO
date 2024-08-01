import multiprocessing
import numpy as np
import mss, keyboard
import time, cv2
import win32gui
import torchvision
import sys
sys.path.insert(0, '.')
from utils.imgProcess import imgProcess
import torch
def dealwith(pic):
    # cv2.imshow('pic',pic)
    # cv2.waitKey(0)

    return imgProcess(pic)

class KeyboardScreenshot:
    def __init__(self):
        self.exit_event = multiprocessing.Event()
        self.queue = multiprocessing.Queue(1)
        self._suskeyboard = multiprocessing.Process(target=self.keyboard_press,args=(self.queue, self.exit_event))
        
    def start(self):
        self._suskeyboard.start()
        
    
    def stop(self):
        self.exit_event.set()


    def keyboard_press(self, queue, event):
        
        while not event.is_set():
            try:
                key = queue.get()
                if key in ['space','shift']:
                    keyboard.press(key)
                    keyboard.release(key)
                else:
                    keyboard.press(key)

            except KeyboardInterrupt:
                print(f"-----stoping-----")
                # self.stop()
                break
    def snapandanalsis(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            screensnap_np = np.array(screenshot)

        return dealwith(screensnap_np)

    def execute_keyboard_down(self, act):
        '''
            STATIC 
                Ascii - KEYBOARD
                turning Agent predict action to Action
        '''
        
        self.queue.put(act)
        
        return self.snapandanalsis()

def set_window_top(window_title):
    hwnd =  win32gui.FindWindow(None, window_title)
    if hwnd !=0:
        win32gui.SetForegroundWindow(hwnd)
if __name__ == '__main__':
    set_window_top("原神")
    key = KeyboardScreenshot()
    key.start()
    for i in ['w','a','d','w','s']:
        key.execute_keyboard_down(i)
        # sustain time
        time.sleep(2)

    time.sleep(3)
    key.stop()
    key._suskeyboard.join()
