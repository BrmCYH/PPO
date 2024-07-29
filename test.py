import threading
import keyboard
import mss
import time
import numpy as np
import cv2, base64
class KeyboardListener:
    def __init__(self):
        self.exit_event = threading.Event()
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener_thread)
        self.screenshot_thread = threading.Thread(target=self.take_screenshot_thread)

    def start(self):
        self.keyboard_thread.start()
        self.screenshot_thread.start()

    def stop(self):
        self.exit_event.set()
        self.keyboard_thread.join()
        self.screenshot_thread.join()

    def keyboard_listener_thread(self):
        def on_key_event(event):
            if event.event_type == keyboard.KEY_DOWN:
                print(f"Key {event.name} was pressed")
                # 在这里可以添加根据按键执行的逻辑

        keyboard.on_press(on_key_event)

        while not self.exit_event.is_set():
            time.sleep(0.1)  # 每隔一段时间检查退出事件

        keyboard.unhook_all()

    def take_screenshot_thread(self):
        with mss.mss() as sct:
            while not self.exit_event.is_set():
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                screenshot_np = np.array(screenshot)
                screenshot_cv2 = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode(".png", screenshot_cv2)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # ret, frame = screenshot.read()
                # cv2.imshow('Screenshot', screenshot_cv2)
                # 在这里可以添加处理截图的逻辑，比如保存、显示或者进一步处理

                time.sleep(10)  # 每隔一秒截图一次

if __name__ == "__main__":
    keyboard_listener = KeyboardListener()
    keyboard_listener.start()

    try:
        # 模拟主程序持续运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, stopping threads...")
        keyboard_listener.stop()
        print("Threads stopped. Exiting.")
