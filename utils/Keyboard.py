import multiprocessing
import numpy as np
import mss, keyboard
import time, cv2
import win32gui
import torchvision
from imgProcess import imgProcess
import torch
def dealwith(pic):
    pointer_dir, target_dir, distance, img_binary = imgProcess(pic)
    img_state = torchvision.transforms.Normalize(0.5,0.5)(img_binary)
    return pointer_dir, target_dir, distance, img_state
    
class KeyboardScreenshotThread:
    def __init__(self):
        self.exit_event = multiprocessing.Event()
        self.queue = multiprocessing.Queue(1)
        self._suskeyboard = multiprocessing.Process(target=self.keyboard_press,args=(self.queue, self.exit_event))
        
    def start(self):
        self._suskeyboard.start()
        
    
    def stop(self):
        self.exit_event.set()
        self._suskeyboard.join()

    def keyboard_press(self, queue, event):
        
        while not event.is_set():
            try:
                key = queue.get()
                keyboard.press(key)
            except KeyboardInterrupt:
                print(f"-----stoping-----")
                break
    def execute_keyboard_down(self, act):
        '''
            STATIC 
                Ascii - KEYBOARD
                turning Agent predict action to Action
        '''
        
        self.queue.put(act)
        
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            screensnap_np = np.array(screenshot)

        status, rewards, terminal = dealwith(screensnap_np)
        return status, rewards, terminal

def set_window_top(window_title):
    hwnd =  win32gui.FindWindow(None, window_title)
    if hwnd !=0:
        win32gui.SetForegroundWindow(hwnd)
if __name__ == '__main__':
    set_window_top("原神")
    key = KeyboardScreenshotThread()
    key.start()
    for i in ['w','a','d','w','s']:
        key.execute_keyboard_down(i)
        # sustain time
        time.sleep(2)

    time.sleep(3)
    key.stop()