#!/usr/bin/env python3

import os,time
#
import threading,subprocess,cv2
import numpy as np
from datetime import datetime
import pytesseract
config = ("-l vie")
from ppadb.client import Client
# import pyautogui
adb = Client(host='127.0.0.1', port=5037)
devices = adb.devices()

if len(devices) == 0:
    print('no device attached')
    quit()

device = devices[0]
print('device:', device.serial)

image = device.screencap()

with open('screen.png', 'wb') as f:
    f.write(image)
    print(os.path.abspath('screen.png'))

class Auto:
    def __init__(self,handle):
        self.handle = handle
        self.count = 0
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

    def patrol_check(self):
        '''
        patrol_check
        pip install pytesseract
        export TESSDATA_PREFIX=/home/nhinhlt/Downloads/tessdata-main
        :return:
        '''
        from editdistance import distance
        screen = self.screen_capture()
        # screen_item_crop = screen[1110 - 0:1275 + 0, 440 - 0:630 + 0]
        screen_name_crop = screen[855 - 0:888 + 0, 83 - 0:636 + 0]
        # screen_name_crop = screen[1295 - 0:1335 + 0, 110 - 0:955 + 0]
        item_name = pytesseract.image_to_string(screen_name_crop, config=config)
        item_name = item_name.strip()

        item_name_short = item_name[item_name.find('tra:') + 4:].strip().replace(':', '').replace('-', '').replace('.',
                                                                                                                   '')
        if 'Tuân' in item_name_short or 'thưởng' in item_name_short:
            item_name_short = item_name[item_name.find('ra:') + 3:].strip().replace(':', '').replace('-', '').replace(
                '.', '')
        list_key = ['Vương miện', 'Pha lê', 'Sừng', 'Quyền trượng', 'Chén thánh', 'Huân chương',
                    '24 Giờ',
                    'Phal&t', 'PhaslG', 'Ouyên trướnG', 'Phalê'
                    ]
        for k in list_key:
            if k in item_name:
                print('patrol:', item_name)
                return True
            elif distance(item_name_short[:len(k)], k) < len(k)/3:
                print('patrol:', item_name)
                return True
        print('skip:', item_name, item_name_short)
        return False

    def patrol_check_color(self, colors = ['whi', 'org']):
        '''
        :return:
        '''

        def check_orange(img):

            hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            cv2.imwrite('debug/{}_imgs_crop.png'.format(self.count), img)

            org_lower = np.array([0, 0, 0], np.uint8)  # orange 0 -20
            org_upper = np.array([20, 255, 255], np.uint8)
            org_mask = cv2.inRange(hsvFrame, org_lower, org_upper)

            # # Make all pixels in mask white
            img[org_mask>0] = [255,255,255]
            img = cv2.bitwise_and(img, img, mask=org_mask)
            cv2.imwrite('debug/{}_imgs_remove_org.png'.format(self.count), img)
            self.count +=1
            if np.sum(org_mask) > org_mask.shape[0] * org_mask.shape[1] * 0.7 * 255:
                return True
            return False


        def check_white(img, thrsh = 200):
            hsvFrame  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

            org_lower = np.array([0, 0, thrsh], np.uint8) #orange 0 -20
            org_upper = np.array([255, 255, 255], np.uint8)
            org_mask = cv2.inRange(hsvFrame, org_lower, org_upper)

            # # Make all pixels in mask white
            # img[org_mask>0] = [255,255,255]
            # img = cv2.bitwise_and(img, img, mask=org_mask)
            # cv2.imwrite('{}_imgs_remove_white.png'.format(self.count), img)
            self.count +=1
            if np.sum(org_mask) > org_mask.shape[0]*org_mask.shape[1]*(0.3 if thrsh<200 else 0.5)*255:
                return True
            return False

        screen = self.screen_capture()

        if screen.shape[0] > 2000:
            screen_item_edge_crop = screen[1118:1250, 451:460]
        else:
            screen_item_edge_crop = screen[742:823, 300:305]

        if screen.shape[0] > 2000:
            screen_item_edge_crop_white = screen[1120:1140, 480:483]
        else:
            screen_item_edge_crop_white = screen[741:755, 320:323]

        if 'whi' in colors and check_white(screen_item_edge_crop_white, thrsh=180 if screen.shape[0] < 2000 else 200):
            print('confirm white')
            return True
        elif 'org' in colors and check_orange(screen_item_edge_crop):
            print('confirm orange')
            return True

        return False

    def improve_general_check(self, require=['politics', 'attack', 'defense', 'leader']):
        '''

        :return:
        '''

        def check_green(img):

            hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            red_lower = np.array([0, 0, 0], np.uint8)  # red 0 -10
            red_upper = np.array([10, 255, 255], np.uint8)
            red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

            green_lower = np.array([50, 0, 0], np.uint8)  # green 50 -60
            green_upper = np.array([60, 255, 255], np.uint8)
            green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

            if np.sum(green_mask) > np.sum(red_mask):
                return True
            return False

        screen = self.screen_capture()
        leader_crop = screen[461:495, 925:1010]
        attack_crop = screen[540:574, 925:1010]
        defense_crop = screen[620:652, 925:1010]
        politics_crop = screen[698:732, 925:1010]

        politics_add = check_green(politics_crop)
        attack_add = check_green(attack_crop)
        defense_add = check_green(defense_crop)
        leader_add = check_green(leader_crop)
        #
        confirm = True
        if 'politics' in require and not politics_add:
            confirm = False
        elif 'politics' in require and not attack_add:
            confirm = False
        elif 'defense' in require and not defense_add:
            confirm = False
        elif 'leader' in require and not leader_add:
            confirm = False
        # if confirm:
        #     print('confirm add', leader_add, attack_add, defense_add, politics_add, )
        # else:
        #     print('skip add', leader_add, attack_add, defense_add, politics_add, )
        return confirm

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

        email = self.file.split("|")[0]
        pwd = self.file.split("|")[1]
        #i = self.index
        device = self.device
        print('device:', device)
        d = Auto(device)
        def click_on_close(d):
            c = 0
            while True:
                try:
                    c += 1
                    print('while continue return', c)
                    poin  = d.find('photos/return.png')
                    # poin  = d.find('photos/x_close.png')
                    if poin > [(0, 0)] :
                        print('poin click on:', poin)
                        d.click(poin[0][0] + 10,poin[0][1] + 10)
                        break
                    time.sleep(1)
                    if c == 10:
                        print('close', c)
                        break
                except Exception as e:
                    print('Exception:', e)
                    return 0
        def patrol_function(d):
            c = 0
            st_time = time.time()
            screen = d.screen_capture()
            if screen.shape[0] > 2000:
                patrolx, patroly = 777, 2250
                refreshx, refreshy = 300, 2250
                nonex, noney = 300, 250
            else:
                patrolx, patroly = 522, 1549
                refreshx, refreshy = 212, 1549
                nonex, noney = 522, 1394
                colors = ['org']
            count = 0
            cost_diamon = 10000
            cost_gold = 10000000

            while True:
                try:
                    c += 1
                    print('diamon:', count, 'gold:', (c - count), 'all:', c, (time.time()-st_time)/60)
                    # patrol_ok = d.patrol_check()
                    patrol_ok = d.patrol_check_color(colors = colors)
                    if patrol_ok:
                        d.click(patrolx,patroly)
                        count +=1
                        time.sleep(1)
                        d.click(nonex,noney)
                    else:
                        d.click(refreshx,refreshy)
                    time.sleep(5)
                    if count*30 > cost_diamon or (c - count)*10000 > cost_gold:
                        click_on_close(d)
                        break
                except Exception as e:
                    print('Exception:', e)
                    return 0

        def impove_general(d):
            c = 0
            confirmx, confirmy = 777, 2250
            refreshx, refreshy = 300, 2250
            nonex, noney = 300, 250
            count = 0
            cost_gold = 1000000

            while True:
                try:
                    c += 1
                    print('gold:', count, 'all:', c)
                    patrol_ok = d.improve_general_check()
                    if patrol_ok:
                        d.click(confirmx, confirmy)
                        count +=1
                    else:
                        d.click(refreshx,refreshy)
                    time.sleep(1)
                    if c * 6000 > cost_gold:
                        click_on_close(d)
                        break
                except Exception as e:
                    print('Exception:', e)
                    return 0

        patrol_function(d)
        # impove_general(d)
        # click_on_close(d)

# GetDevices()
thread_count = 1


def strew():
    for m in range(thread_count):
        threading.Thread(target=main, args=(m,)).start()

def main(m):
        # device = GetDevices()[m]
        print('device:', device.serial)
        tk = ['nhinhlt@gmail.com|nhinh2202']
        for i in range(m, len(tk), thread_count):
                mail = tk[i].strip()
                run = starts(device,mail, device,)
                run.run()

strew()