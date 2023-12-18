#!/usr/bin/env python3
# import cv2
import os,time
import threading
import subprocess
import base64
import random
import requests
from datetime import datetime
# import pyautogui
import cv2
import numpy as np

from ppadb.client import Client
adb = Client(host='127.0.0.1', port=5037)
devices = adb.devices()
if len(devices) == 0:
    print('no device attached')
    quit()

def get_screen():
    for i, device in enumerate(devices):
        image = devices[i].screencap()
        with open('photos/{}_{}_screen_add_mark.png'.format(i, devices[i].serial), 'wb') as f:
            f.write(image)

if __name__ == '__main__':
    get_screen()
    '''
    python get_screen.py
    '''