'''
    Manipulat 
        keyboard
        snapshot
        
'''


import numpy as np
import base64

import keyboard
import threading
class Interface:
    def __enter__(self):
        self._hook = keyboard(self._on_key_event)
    pass
    def __exit__(self,):
        keyboard.un

            