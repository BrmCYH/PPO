import threading
import numpy as np
import mss, keyboard
import time, cv2
import pyautogui
def dealwith(pic):
    
    return 1,1,1
class KeyboardScreenshotThread:
    def __init__(self):
        self.exit_event = threading.Event()
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener_thread)
        
    def start(self):
        self.keyboard_thread.start()
        
    
    def stop(self):
        self.exit_event.set()
        self.keyboard_thread.join()
    
    def keyboard_listener_thread(self):
        def on_key_event(event):
            if event.event_type == keyboard.KEY_DOWN:
                print(f"Key {event.name} was pressed")
                self.take_screenshot = True
        
        keyboard.on_press(on_key_event)

        while not self.exit_event.is_set():
            time.sleep(0.1)  # 每隔一段时间检查退出事件

        keyboard.unhook_all()
    def execute_keyboard_down(self, act):
        '''
            STATIC 
                Ascii - KEYBOARD
                turning Agent predict action to Action
        '''
        # Press keyboard
        pyautogui.press('space')

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            screensnap_np = np.array(screenshot)

        status, rewards, terminal = dealwith(screensnap_np)
        return status, rewards, terminal

if __name__ == "__main__":
    thread_manager = KeyboardScreenshotThread()
    thread_manager.start()

    # 主程序可以继续执行其他任务
    try:
        i=0
        while True:
            time.sleep(1)  # 这里可以加入主程序的逻辑
            x = time.time()
            i=i+1
            print(i)
            print(thread_manager.execute_keyboard_down(1),f"time executed {time.time()-x}")
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, stopping threads...")
        thread_manager.stop()
        print("Threads stopped.")