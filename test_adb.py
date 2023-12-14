#!/usr/bin/env python3

from ppadb.client import Client
from PIL import Image
import numpy
import time
import pyautogui
adb = Client(host='127.0.0.1', port=5037)
devices = adb.devices()

if len(devices) == 0:
    print('no device attached')
    quit()

device = devices[0]
print('device:', device)

# while True:
#     image = device.screencap()
#
#     with open('screen.png', 'wb') as f:
#         f.write(image)
#     break

import os,time
#
import threading,subprocess,base64,cv2,random,requests,pyautogui
import numpy as np
from datetime import datetime

class Auto:
    def __init__(self,handle):
        self.handle = handle
    def screen_capture(self):
        image_bytes = device.screencap()
        image = cv2.imdecode(np.asarray(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        #
        # pipe = subprocess.Popen("adb shell screencap -p",
        #                         stdin=subprocess.PIPE,
        #                         stdout=subprocess.PIPE, shell=True)
        # image_bytes = pipe.stdout.read().replace(b'\r\n', b'\n')
        # image = cv2.imdecode(np.fromstring(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        #
        # pipe = subprocess.Popen(f'adb -s {self.handle} exec-out screencap -p',
        #                 stdin=subprocess.PIPE,
        #                 stdout=subprocess.PIPE, shell=True)
        # image_bytes = pipe.stdout.read()
        # image = cv2.imdecode(np.fromstring(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        return image
    def click(self,x,y):
        # os.system(f"adb -s {self.handle} shell input tap {x} {y}")
        device.shell(f'input tap {x} {y}')

    def delete(self,package):
        device.shell(f"shell pm clear {package} ")
    def off(self,package):
        device.shell(f"shell am force-stop {package} ")

    def find(self,img='',threshold=0.99):
        img = cv2.imread(img)
        img2 = self.screen_capture()
        result = cv2.matchTemplate(img,img2,cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        retVal = list(zip(*loc[::-1]))
        #image = cv2.rectangle(img2, retVal[0],(retVal[0][0]+img.shape[0],retVal[0][1]+img.shape[1]), (0,250,0), 2)
        #cv2.imshow("test",image)
        #cv2.waitKey(0)
        #cv2.destroyWindow("test")
        return retVal
def GetDevices():
        devices = subprocess.check_output("adb devices")
        p = str(devices).replace("b'List of devices attached","").replace('\\r\\n',"").replace(" ","").replace("'","").replace('b*daemonnotrunning.startingitnowonport5037**daemonstartedsuccessfully*Listofdevicesattached',"")
        if len(p) > 0:
            listDevices = p.split("\\tdevice")
            listDevices.pop()
            return listDevices
        else:
            return 0

class starts(threading.Thread):
    def __init__(self, nameLD,file, devicee):
        super().__init__()
        self.nameLD = nameLD
        self.file = file
        self.device = devicee
    def run(self):
        print('==================================================')

        email = self.file.split("|")[0]
        pwd = self.file.split("|")[1]
        #i = self.index
        device = self.device
        print('device:', device)
        d = Auto(device)
        def step1(d):
            c = 0
            while True:
                try:
                    c += 1
                    print('while continue', c)
                    poin  = d.find('/home/nhinhlt/project/nhinhlt/python_adb/Photos-001/x_close.png')
                    if poin > [(0, 0)] :
                        print('poin click on:', poin)
                        d.click(poin[0][0],poin[0][1])
                        # print(" \033[1;31m |\033[1;37m[",self.nameLD,"]\033[1;31m Mở Face | Time:", time.ctime(time.time()))
                        #
                        step2(d)
                        break
                    time.sleep(3)
                    if c == 10:
                        print('close', c)
                        d.off('com.evony.katana')
                        break
                except Exception as e:
                    print('Exception:', e)
                    return 0
        def click_on_iron16(d):
            c = 0
            while True:
                try:
                    c += 1
                    print('while continue', c)
                    poin  = d.find('/home/nhinhlt/project/nhinhlt/python_adb/Photos-001/iron16.png', threshold=0.5)
                    if poin > [(0, 0)] :
                        print('poin click on iron 16:', poin)
                        d.click(poin[0][0] + 20,poin[0][1] + 100)
                        step2(d)
                        # break
                    time.sleep(3)
                    if c == 10:
                        print('close', c)
                        # d.off('com.evony.katana')
                        break
                except Exception as e:
                    print('Exception:', e)
                    return 0
        def step2(d):
            print('close')
            # d.delete('com.instagram.android')
        # step1(d)
        click_on_iron16(d)

# GetDevices()
thread_count = 1


def strew():
    # pyautogui.doubleClick('/home/nhinhlt/project/nhinhlt/python_adb/Photos-001/icon_open_evony2.png')
    # time.sleep(5)
    # print('----------------------------------done-----------------------')
    # exit(0)
    # pyautogui.click('/home/nhinhlt/project/nhinhlt/python_adb/Photos-001/x_close.png')
    # time.sleep(30)
    # # GetDevices()
    for m in range(thread_count):
        threading.Thread(target=main, args=(m,)).start()

def main(m):
        # device = GetDevices()[m]
        print('device:', device)
        tk = ['nhinhlt@gmail.com|nhinh2202']
        for i in range(m, len(tk), thread_count):
                mail = tk[i].strip()
                run = starts(device,mail, device,)
                run.run()

strew()
# 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
###  [(405, 1096), (406, 1096), (405, 1097), (406, 1097), (405, 1098), (406, 1098), (405, 1099), (406, 1099)]