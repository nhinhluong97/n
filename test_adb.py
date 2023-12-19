#!/usr/bin/env python3
import cv2
import os,time
import threading
import subprocess
import base64
import random
import requests
from datetime import datetime
# import pyautogui
import sys
import numpy as np

from ppadb.client import Client
os.system("adb devices")
adb = Client(host='127.0.0.1', port=5037)
devices = adb.devices()
if len(devices) == 0:
    print('no device attached')
    quit()
os.system("export TESSDATA_PREFIX=/home/nhinhlt/Downloads/tessdata-main")

# exit(0)
class Auto:
    def __init__(self,handle, device_id):
        self.handle = handle
        self.device_id = device_id
        self.count = 0
    def screen_capture(self):
        image_bytes = devices[self.device_id].screencap()
        image = cv2.imdecode(np.asarray(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return image
    def click(self,x,y):
        os.system(f"adb -s {self.handle} shell input tap {x} {y}")
        # device.shell(f'input tap {x} {y}')

    def wipe(self,x, y, x2, y2):
        # os.system(f"adb -s {self.handle} shell input touchscreen swipe {x}, {y}, {x2}, {y2}")
        os.system(f"adb -s {self.handle} shell input touchscreen swipe {x} {y} {x2} {y2}")

    def delete(self,package):
        os.system(f"adb -s {self.handle} shell pm clear {package} ")
        # device.shell(f"shell pm clear {package} ")
    def off(self,package):
        os.system(f"adb -s {self.handle} shell am force-stop {package} ")

    def find(self,img='',threshold=0.99):
        img = cv2.imread(img)
        img2 = self.screen_capture()
        result = cv2.matchTemplate(img,img2,cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        # return [0,0]
        retVal = list(zip(*loc[::-1]))
        #image = cv2.rectangle(img2, retVal[0],(retVal[0][0]+img.shape[0],retVal[0][1]+img.shape[1]), (0,250,0), 2)
        #cv2.imshow("test",image)
        #cv2.waitKey(0)
        #cv2.destroyWindow("test")
        return retVal, img.shape[1]//2, img.shape[0]//2

    def patrol_check_color(self, colors = ['whi', 'org']):
        '''
        :return:
        '''

        def check_orange(img):

            hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            org_lower = np.array([0, 0, 0], np.uint8)  # orange 0 -20
            org_upper = np.array([20, 255, 255], np.uint8)
            org_mask = cv2.inRange(hsvFrame, org_lower, org_upper)

            if np.sum(org_mask) > org_mask.shape[0] * org_mask.shape[1] * 0.7 * 255:
                return True
            return False


        def check_white(img, size=2000):

            hsvFrame  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

            org_lower = np.array([0, 0, 180], np.uint8) #orange 0 -20
            org_upper = np.array([255, 255, 255], np.uint8)
            org_mask = cv2.inRange(hsvFrame, org_lower, org_upper)

            # Make all pixels in mask white
            # img[org_mask>0] = [255,255,255]
            # img = cv2.bitwise_and(img, img, mask=org_mask)
            # cv2.imwrite('debug/{}_imgs_remove_white.png'.format(self.count), img)
            self.count +=1
            if np.sum(org_mask) > org_mask.shape[0]*org_mask.shape[1]*(0.3 if size<2000 else 0.5)*255:
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

        if 'whi' in colors and check_white(screen_item_edge_crop_white, size=2000):
        # if 'whi' in colors and check_white(screen_item_edge_crop_white, thrsh=180 if screen.shape[0] < 2000 else 200):
            print('confirm white')
            return True
        elif 'org' in colors and check_orange(screen_item_edge_crop):
            print('confirm orange')
            return True

        return False

    def check_rally_curr(self, rally_coodinates_dict={}):
        '''
        :return:
        '''

        def check_color(img, lower_h=0, upper_h=10, lower_v=0, upper_v=255,
                        lower_s=0, upper_s=255, thrsh=0.5, debug=False):
            '''
            red 0-10
            green 50-60
            orange 0-20
            white 0-255, 0-255, 180-255
            :param img:
            :param lower_h:
            :param upper_h:
            :param lower_v:
            :return:
            '''

            hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower = np.array([lower_h, lower_s, lower_v], np.uint8)
            upper = np.array([upper_h, upper_s, upper_v], np.uint8)
            mask = cv2.inRange(hsvFrame, lower, upper)
            if debug:
                # Make all pixels in mask white
                img[mask > 0] = [255, 255, 255]
                # img = cv2.bitwise_and(img, img, mask=mask)
                cv2.imwrite('debugs/{}_imgs_remove_{}_{}_{}.png'.format(self.count, lower_h, upper_h, lower_v), img)
                self.count +=1
            if np.sum(mask) > mask.shape[0] * mask.shape[1] * thrsh * 255:
                return True
            return False

        screen = self.screen_capture()

        if screen.shape[0] > 2000:
            x1, y1, x2, y2 = rally_coodinates_dict[2000]['screen_rally1']
            rally_bock_h = rally_coodinates_dict[2000]['rally_bock_h']
        else:
            x1, y1, x2, y2 = rally_coodinates_dict[1600]['screen_rally1']
            rally_bock_h = rally_coodinates_dict[1600]['rally_bock_h']

        screen_rally1 = screen[y1:y2, x1:x2]
        screen_rally2 = screen[y1 + rally_bock_h:y2 + rally_bock_h, x1:x2]  # 1623 - 982
        screen_rally3 = screen[y1 + rally_bock_h * 2:y2 + rally_bock_h * 2, x1:x2]

        # cv2.imwrite('debugs/screen_rally1.png', screen_rally1)
        # cv2.imwrite('debugs/screen_rally2.png', screen_rally2)
        # cv2.imwrite('debugs/screen_rally3.png', screen_rally3)
        confirm = []

        rally_time_crop3 = screen_rally3[screen_rally3.shape[0] // 2:].copy()
        rally3 = check_color(screen_rally3, lower_h=50, upper_h=60, thrsh=0.7, lower_s=80, upper_s=255, lower_v=0,
                             upper_v=150)
        rally_time3 = check_color(rally_time_crop3, lower_h=0, upper_h=10, thrsh=0.1)
        if rally3 and not rally_time3:
            confirm.append(3)
            print('confirm rally3', confirm)
            return confirm[0]

        rally_time_crop2 = screen_rally2[screen_rally2.shape[0] // 2:].copy()
        rally2 = check_color(screen_rally2,
                             lower_h=50, upper_h=60, thrsh=0.6, lower_s=80, upper_s=255, lower_v=0, upper_v=150)
        rally_time2 = check_color(rally_time_crop2, lower_h=0, upper_h=10, thrsh=0.1)
        if rally2 and not rally_time2:
            confirm.append(2)
            print('confirm rally2', confirm)
            return confirm[0]
        rally_time_crop1 = screen_rally1[screen_rally1.shape[0] // 2:].copy()
        rally1 = check_color(screen_rally1, lower_h=50, upper_h=60, thrsh=0.7, lower_s=80, upper_s=255, lower_v=0,
                             upper_v=150)
        rally_time1 = check_color(rally_time_crop1,
                                  lower_h=0, upper_h=10, thrsh=.1)
        if rally1 and not rally_time1:
            confirm.append(1)
            print('confirm rally1', confirm)
            return confirm[0]
        return 0

    def check_rally_valid(self, rally_coodinates_dict={}):
        '''
        :return:
        '''

        def check_color(img, lower_h=0, upper_h=10, lower_v=0, upper_v=255,
                        lower_s=0, upper_s=255, thrsh=0.5, debug=False):
            '''
            red 0-10
            green 50-60
            orange 0-20
            white 0-255, 0-255, 180-255
            :param img:
            :param lower_h:
            :param upper_h:
            :param lower_v:
            :return:
            '''

            hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower = np.array([lower_h, lower_s, lower_v], np.uint8)
            upper = np.array([upper_h, upper_s, upper_v], np.uint8)
            mask = cv2.inRange(hsvFrame, lower, upper)
            if debug:
                # Make all pixels in mask white
                img[mask > 0] = [255, 255, 255]
                # img = cv2.bitwise_and(img, img, mask=mask)
                cv2.imwrite('debugs/{}_imgs_remove_{}_{}_{}.png'.format(self.count, lower_h, upper_h, lower_v), img)
                self.count +=1
            if np.sum(mask) > mask.shape[0] * mask.shape[1] * thrsh * 255:
                return True
            return False

        screen = self.screen_capture()
        if screen.shape[0] > 2000:
            x1, y1, x2, y2 = rally_coodinates_dict[2000]['run_button']
        else:
            x1, y1, x2, y2 = rally_coodinates_dict[1600]['run_button']

        screen_run_botton = screen[y1:y2, x1:x2]
        run_valid = check_color(screen_run_botton, lower_h=0, upper_h=20, thrsh=0.7, debug=False)
        if run_valid:
            print('confirm run_valid', run_valid)
            return True

        return False

    def check_general_valid(self, rally_coodinates_dict={}):
        '''
        :return:
        '''

        def check_color(img, lower_h=0, upper_h=10, lower_v=0, upper_v=255,
                        lower_s=0, upper_s=255, thrsh=0.5, debug=True):
            '''
            red 0-10
            green 50-60
            orange 0-20
            white 0-255, 0-255, 180-255
            :param img:
            :param lower_h:
            :param upper_h:
            :param lower_v:
            :return:
            '''

            hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower = np.array([lower_h, lower_s, lower_v], np.uint8)
            upper = np.array([upper_h, upper_s, upper_v], np.uint8)
            mask = cv2.inRange(hsvFrame, lower, upper)
            if debug:
                # Make all pixels in mask white
                img[mask > 0] = [255, 255, 255]
                # img = cv2.bitwise_and(img, img, mask=mask)
                cv2.imwrite('debugs/{}_imgs_remove_{}_{}_{}.png'.format(self.count, lower_h, upper_h, lower_v), img)
                self.count +=1
            if np.sum(mask) > mask.shape[0] * mask.shape[1] * thrsh * 255:
                return True
            return False

        screen = self.screen_capture()
        if screen.shape[0] > 2000:
            x1, y1, x2, y2 = rally_coodinates_dict[2000]['general']
        else:
            x1, y1, x2, y2 = rally_coodinates_dict[1600]['general']

        screen_general = screen[y1:y2, x1:x2]
        general_invalid = check_color(screen_general, lower_h=0, upper_h=30,
                                    lower_s=150, upper_s=255, lower_v=0, upper_v=150, thrsh=0.5, debug=True)
        if not general_invalid:
            print('confirm general_valid', not general_invalid)
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

    def find2(self, img, threshold=0.9):
        '''
        :return:
        '''

        screen = self.screen_capture()

        result = cv2.matchTemplate(img, screen, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        retVal = list(zip(*loc[::-1]))
        # # We want the minimum squared difference
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_loc > (0, 0):
            ret = max_loc[0] + img.shape[1] // 2, max_loc[1] + img.shape[0] // 2
            return ret
        return False

    def mode_color(self, arr, top=3, chanel=3):
        '''
        mode color: (1, 1, 0), 11
        :param arr:
        :return:
        '''
        from collections import Counter
        arr = arr.reshape((-1, chanel))
        l = list(map(tuple, arr))
        # l = list(arr)
        occurence_count = Counter(l)
        res = occurence_count.most_common(top)
        mode = []
        count = 0
        for m, c in res:
            mode.append(list(m))
            # mode.append(m)
            count += c
        return mode, count
    def check_mode_color(self, img, thrsh=0.03, debug=True, color='red', top=5):
        '''
        red 0-10
        green 50-60
        orange 0-20
        white 0-255, 0-255, 180-255
        :param img:
        :param lower_h:
        :param upper_h:
        :param lower_v:
        :return:
        '''
        if isinstance(color, list):
            lower_h, upper_h, lower_v, upper_v, lower_s, upper_s = color
        elif color == 'red':
            lower_h, upper_h = 0, 10
            lower_s, upper_s = 0, 255
            lower_v, upper_v = 0, 150

        elif color == 'green':
            lower_h, upper_h = 50, 70
            lower_s, upper_s = 0, 255
            lower_v, upper_v = 60, 90
        elif color == 'orange':
            lower_h, upper_h = 0, 20
            lower_s, upper_s = 0, 255
            lower_v, upper_v = 0, 255
        elif color == 'white':
            lower_h, upper_h = 0, 255
            lower_s, upper_s = 180, 255
            lower_v, upper_v = 0, 255
        elif color == 'blue':
            lower_h, upper_h = 80, 120
            lower_s, upper_s = 0, 255
            lower_v, upper_v = 0, 150
        else:
            lower_h, upper_h = 0, 255
            lower_s, upper_s = 0, 255
            lower_v, upper_v = 0, 255

        hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsvFrame = np.round(hsvFrame, -1)
        # hFrame, sFrame, vFrame = np.split(hsvFrame,3, axis=2)
        # print('hsvFrame:', hFrame)
        mode, count_mode = self.mode_color(hsvFrame, top, chanel=3)

        mode_arr = np.array([mode], np.uint8)
        # print(mode_arr)

        lower = np.array([lower_h, lower_s, lower_v], np.uint8)
        upper = np.array([upper_h, upper_s, upper_v], np.uint8)
        mask = cv2.inRange(mode_arr, lower, upper)

        if debug:
            print('mode, count:', mode, count_mode, img.shape, count_mode / (img.shape[0] * img.shape[1]))
            print('upper:', upper)
            print('lower:', lower)
            # print('mask:', mask)

        if np.sum(mask) > top * 255 / 2 and count_mode > img.shape[0] * img.shape[1] * thrsh:
            return True
        return False
    def check_box_color(self, box, color='red', name=0, thrsh=0.5, top=5):
        '''
        :return:
        '''
        screen = self.screen_capture()

        x1, y1, x2, y2 = box
        screen_run_botton = screen[y1:y2, x1:x2]

        cv2.imwrite('debugs/check_box_color{}.png'.format(name), screen_run_botton)
        confirm = self.check_mode_color(screen_run_botton, color=color, thrsh=thrsh, top=top)

        if confirm:
            print('confirm {}'.format(color))
        return confirm

    def check_hint(self, box, hint_img, thrsh_and=0.5, thrsh_xor=0.3, name=0, clear_noise=False, binary_thrsh=80):

        screen = self.screen_capture()
        # hint_img = cv2.imread('photos/hint_binary.png', 0)
        x1, y1, x2, y2 = box
        screen_run_botton = screen[y1:y2, x1:x2]
        if clear_noise:
            screen_run_botton[25:60, 30:69] = np.zeros((35, 39, 3))
        gray_img = cv2.cvtColor(screen_run_botton, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
        print(hint_img.shape, binary_img.shape)
        xor_img = cv2.bitwise_xor(hint_img, binary_img)
        and_img = cv2.bitwise_and(hint_img, binary_img)
        total = np.sum(binary_img)
        if np.sum(xor_img) < total * thrsh_xor and np.sum(and_img) > total * thrsh_and:
            # cv2.imwrite('debugs/bitwise_and{}.png'.format(name), and_img)
            # cv2.imwrite('debugs/bitwise_xor{}.png'.format(name), xor_img)
            return True

        return False

    def find_madi(self, mandi_img, rally_coodinates_dict={}, threshold=0.85):
        '''
        :return:
        '''

        def image_2_text(img, l='vie'):
            '''
            export TESSDATA_PREFIX=/home/nhinhlt/Downloads/tessdata-main
            :return:
            '''
            import pytesseract
            # config = ("-l vie")
            config = ("-l {}".format(l))

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img_gray = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)

            img_gray = ~img_gray
            text = pytesseract.image_to_string(img_gray, config=config)
            return text.strip()
        screen = self.screen_capture()

        result = cv2.matchTemplate(mandi_img, screen, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        retVal = list(zip(*loc[::-1]))
        # # We want the minimum squared difference
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # print(max_loc, max_val,threshold)
        if max_loc > (0, 0) and max_val>threshold:
            ret = max_loc[0] + mandi_img.shape[1] // 2, max_loc[1] + mandi_img.shape[0] // 2
            name_box = [ret[0] - 175, ret[1] + 95, ret[0] - 90, ret[1] + 130]
            name = image_2_text(screen[name_box[1]:name_box[3], name_box[0]:name_box[2]], l='eng')
            name = name.replace('|', '').replace(' ', '').lower()
            print('======================================name:', name, max_loc, max_val)

            # image = cv2.rectangle(screen, max_loc, (max_loc[0] + mandi_img.shape[1], max_loc[1] + mandi_img.shape[0]),
            #                       (0, 0, 255), 3)
            # image = cv2.rectangle(image, (name_box[0], name_box[1]), (name_box[2], name_box[3]), (0, 0, 255), 3)
            # cv2.imwrite('./debugs/{}_mandi.png'.format(self.count), image)
            return ret, name
        return False, ''

class starts(threading.Thread):
    def __init__(self, nameLD,file, device_serial, list_task):
        super().__init__()
        self.nameLD = nameLD
        self.file = file
        self.device = device_serial
        self.list_task = list_task
    def run(self):
        email = self.file.split("|")[0]
        pwd = self.file.split("|")[1]

        d = Auto(self.device, self.nameLD)
        print('====================runing==================')
        def click_on_x_icon(d, ret=True, clo=True):
            c = 0
            while True:
                try:
                    c += 1
                    # print('while continue', c)
                    if clo:
                        poin, w, h  = d.find('photos/x_close.png')
                        if poin > [(0, 0)] :
                            print('pclick_on_x_icon:', poin[0][0] + w, poin[0][1] + h)
                            d.click(poin[0][0] + w, poin[0][1] + h)
                            break
                    if ret:
                        d.click(30, 50)
                        break
                        # poin, w, h = d.find('photos/return.png')
                        # if poin > [(0, 0)] :
                        #     print('pclick_on_return_icon:', poin[0][0] + w, poin[0][1] + h)
                        #     d.click(poin[0][0] + w, poin[0][1] + h)
                        #     break
                    time.sleep(1)
                    if c == 10:
                        # print('close', c)
                        break
                except Exception as e:
                    print('Exception on click_on_x_icon:', e)
                    return 0

        def patrol_function(d, sleep=[1, 3], colors = ['whi', 'org']):
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
                # colors = ['org']
            count = 0
            cost_diamon = 10000
            cost_gold = 10000000

            while True:
                try:
                    c += 1
                    print('diamon:', count, 'gold:', (c - count), 'all:', c, 'time:', (time.time() - st_time)/(60))
                    # patrol_ok = d.patrol_check()
                    patrol_ok = d.patrol_check_color(colors=colors)
                    if patrol_ok:
                        d.click(patrolx,patroly)
                        count +=1
                        time.sleep(sleep[0])
                        d.click(nonex,noney)
                    else:
                        d.click(refreshx,refreshy)
                    time.sleep(sleep[1])
                    if count*30 > cost_diamon or (c - count)*10000 > cost_gold:
                        click_on_x_icon(d)
                        break
                except Exception as e:
                    print('Exception:', e)
                    return 0

        def join_rally_function(d, mode='auto'):
            c = 0
            count = 0
            cost_saitama = 3000
            screen_size = d.screen_capture().shape[0]
            screen_size = (screen_size//1000)*1000
            rally_coodinates_dict = {2000: {'screen_rally1': [806, 992, 1024, 1058], 'rally_bock_h': 636,
                                            'run_button':[575, 2219, 971, 2302], 'reset_botton':[120, 2219, 482, 2302],
                                            'add_1_best':[734, 1759, 753, 1782],
                                            'troop1':[53, 250, 162, 350],
                                            'troop2':[183, 250, 285, 350],
                                            'troop3':[306, 250, 400, 350],
                                            'troop4':[429, 250, 526, 350],
                                            'general':[74, 1054, 276, 1252],
                                            'none':[500, 50],
                                            },
                                     1600: {'screen_rally1': [806, 992, 1024, 1058], 'rally_bock_h': 230,
                                            'run_button':[575, 2219, 971, 2302],'reset_botton':[120, 2219, 482, 2302]}
                                     }
            while True:
                try:
                    x1, y1 = rally_coodinates_dict[screen_size]['none']
                    d.click(x1, y1)

                    c += 1
                    print('saitame:', count*30, 'number rally:', count, 'scan:', c)
                    ### check ralling exists
                    rally_curr = d.check_rally_curr(rally_coodinates_dict)
                    if rally_curr:
                        x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['screen_rally1']
                        block_h = rally_coodinates_dict[screen_size]['rally_bock_h']
                        ### click join
                        d.click((x1 + x2)//2, (y1 + y2)//2 + block_h*(rally_curr-1))
                        # time.sleep(1)
                        run_valid = d.check_rally_valid(rally_coodinates_dict)
                        if run_valid:
                            if mode=='auto':
                                ###### auto general, +1 best
                                x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['reset_botton']
                                d.click((x1 + x2)//2, (y1 + y2)//2)
                                # time.sleep(1)
                                x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['add_1_best']
                                d.click((x1 + x2)//2, (y1 + y2)//2)
                                # time.sleep(1)
                                x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['run_button']
                                d.click((x1 + x2)//2, (y1 + y2)//2)
                            else:
                                ###### sellect troop
                                for troopid in ['troop1', 'troop2', 'troop4', ]:
                                    x1, y1, x2, y2 = rally_coodinates_dict[screen_size][troopid]
                                    d.click((x1 + x2)//2, (y1 + y2)//2)
                                    # time.sleep(1)
                                    valid = d.check_general_valid(rally_coodinates_dict)
                                    if valid:
                                        print('troopid', troopid)
                                        break
                                else:
                                    print('troop default')

                                x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['run_button']
                                d.click((x1 + x2)//2, (y1 + y2)//2)
                            count +=1
                        pass
                    else:
                        time.sleep(60)

                    if count*30 > cost_saitama:
                        click_on_x_icon(d)
                        break
                except Exception as e:
                    print('Exception:', e)
                    return 0

        def say_hi(d, hi_time = 10):
            rally_coodinates_dict = {2000: {'hi_botton': [62, 424, 206, 559],
                                            'none': [800, 1300],
                                            'wipe': [564, 806, 588, 1380],
                                            },
                                     }

            c = 0
            screen_size = d.screen_capture().shape[0]
            screen_size = (screen_size//1000)*1000

            while True:
                try:
                    c += 1
                    print('say hi:', c, )
                    x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['hi_botton']
                    d.click((x1 + x2)//2, (y1 + y2)//2)
                    time.sleep(1)
                    print('hi')
                    x1, y1 = rally_coodinates_dict[screen_size]['none']
                    d.click(x1, y1)
                    time.sleep(1)
                    d.click(x1, y1)
                    time.sleep(1)
                    d.click(x1, y1)
                    time.sleep(1)
                    d.click(x1, y1)
                    time.sleep(1)
                    d.click(x1, y1)
                    time.sleep(60)
                    if c >= hi_time:
                        print('--------hi done--------')
                        # click_on_x_icon(d)
                        break
                except Exception as e:
                    print('Exception:', e)
                    return 0

        def wipe_she(d, wipe_time = 15):
            rally_coodinates_dict = {2000: {'hi_botton': [62, 424, 206, 559],
                                            'none': [800, 1300],
                                            'wipe': [564, 806, 588, 1380],
                                            },
                                     }
            c = 0


            screen_size = d.screen_capture().shape[0]
            screen_size = (screen_size//1000)*1000
            while True:
                try:
                    c += 1
                    print('wipe:', c, )
                    x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['wipe']
                    d.wipe(x1, y1, x2, y2)
                    time.sleep(3)
                    if c >= wipe_time:
                        # click_on_x_icon(d)
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
                        click_on_x_icon(d)
                        break
                except Exception as e:
                    print('Exception:', e)
                    return 0

        def find_mandi(d, cost_distance_x=1000, cost_distance_y=200, thrsh=0.8):
            c = 0
            count = 0
            # cost_distance_x = 0
            # cost_distance_y = 50
            screen_size = d.screen_capture().shape[0]
            screen_size = (screen_size//1000)*1000
            print('screen_size:', screen_size)

            rally_coodinates_dict = {2000: {
                'wipe_up': [964, 940, 100, 1400],
                'wipe_down': [100, 1400, 964, 940],
                'wipe_left': [155, 940, 964, 1400],
                'wipe_right': [964, 1400, 155, 940],
                'add_botton':[295, 1476, 766, 1558],
                                                }
                                    }
            mandi_img = cv2.imread('photos/mandi1.png')
            mark_icon = cv2.imread('photos/add_mark.png')
            x_c = 0
            x_step = 10
            y_c = 0
            y_step = 10
            count = 0
            while True:
                ''' move right'''
                # try:
                if True:
                    print('distance x:', x_c*x_step if y_c%2==0 else cost_distance_x - x_c*x_step,
                          'distance y:', y_c*y_step, 'found:', count)
                    x_c += 1
                    ### check ralling exists
                    st_find = time.time()
                    mandi, name = d.find_madi(mandi_img, rally_coodinates_dict, thrsh)
                    # print('find time:', time.time() - st_find)
                    if mandi and name not in ['cap1', 'cap2', 'cap3', 'cap5', 'caps','cans', ]:
                        ## click mandi
                        d.click(mandi[0], mandi[1])
                        # time.sleep(1)
                        st_find_mark = time.time()
                        mark_point = d.find2(mark_icon)
                        # print('st_find_mark time:', time.time() - st_find_mark)

                        ### click add
                        d.click(mark_point[0], mark_point[1])
                        ### click firm
                        x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['add_botton']
                        d.click((x1+x2)//2, (y1+y2)//2)
                        count +=1
                    print('a step time:', time.time() - st_find)

                    if x_c*x_step > cost_distance_x:
                        # move down
                        x1, y1, x2, y2 =rally_coodinates_dict[screen_size]['wipe_down']
                        d.wipe(x1, y1, x2, y2)
                        time.sleep(3)
                        print('distance y:', y_c*y_step)
                        y_c +=1
                        x_c = 0
                    elif y_c%2==0:
                        x1, y1, x2, y2 =rally_coodinates_dict[screen_size]['wipe_right']
                        d.wipe(x1, y1, x2, y2)
                        time.sleep(2)
                    else:
                        x1, y1, x2, y2 =rally_coodinates_dict[screen_size]['wipe_left']
                        d.wipe(x1, y1, x2, y2)
                        time.sleep(2)
                    if y_c*y_step > cost_distance_y:
                        break
                # except Exception as e:
                #     print('Exception:', e)
                #     return 0

        def attact_monster(d):
            screen_size = d.screen_capture().shape[0]
            screen_size = (screen_size//1000)*1000
            print('screen_size:', screen_size)
            rally_coodinates_dict = {2000: {

                                        'runing_btn1': [530, 415, 601, 463],
                                        'runing_space': 70 + 5,

                                        'list_mark_btn': [28, 1863, 155, 1964],
                                        'central_box': [373, 1130, 691, 1300],
                                        'first_monster': [55, 507, 605, 677],
                                        'delete_btn': [621, 610, 729, 688],
                                        'firm_delete_btn': [579, 1358, 853, 1426],
                                        'cancel_delete_btn': [226, 1358, 497, 1426],

                                        'attack_monster_btn': [131, 943, 306, 1085],
                                        'occupy_btn': [306, 806, 466, 949],
                                        'hint1': [164, 858, 203, 896],  # kiếm chéo
                                        'hint2': [51, 800, 146, 891],  # level box
                                        'attack_btn_only': [353, 1458, 719, 1525],

                                        'attack_btn': [110, 1768, 474, 1837],
                                        'war_btn': [595, 1768, 966, 1837],
                                        'exit_btn': [1000, 577, 1060, 639],
                                        'attack_btn2': [110, 1938, 474, 2005],
                                        'war_btn2': [595, 1938, 966, 2005],
                                        'warning_firm_btn': [577, 1358, 853, 1420],
                                        'warning_cancel_btn': [226, 3358, 497, 1420],

                                        'run_button': [575, 2219, 971, 2302], 'reset_botton': [120, 2219, 482, 2302],
                                        'add_1_best': [734, 1759, 753, 1782],
                                        'troop1': [53, 250, 162, 350],
                                        'troop2': [183, 250, 285, 350],
                                        'troop3': [306, 250, 400, 350],
                                        'troop4': [429, 250, 526, 350],
                                        'general': [77, 546, 268, 739],
                                        'wipe_none': [595, 1768, 966, 1837],
                                                }
                                    }
            hint_monster_img = cv2.imread('photos/hint_binary.png', 0)
            hint_invalid_general_img = cv2.imread('photos/hint_invalid_general_binary.png', 0)
            hint_monster_img = np.array(hint_monster_img, np.uint8)
            hint_invalid_general_img = np.array(hint_invalid_general_img, np.uint8)

            troop_valid_number = 3
            x1, y1, x2, y2 = box = rally_coodinates_dict[screen_size]['runing_btn1']  # blue thresh 0.1 top 5
            rally_coodinates_dict[screen_size]['check_runing_btn'] = [x1, y1 + (troop_valid_number - 1) * rally_coodinates_dict[screen_size]['runing_space'], x2,
                   y2 + (troop_valid_number - 1) * rally_coodinates_dict[screen_size]['runing_space']]
            c = 0
            while True:
                # try:
                x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['check_runing_btn']
                troop_valid = not d.check_box_color([x1, y1, x2, y2], color='blue', thrsh=0.1)
                if troop_valid:
                    x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['list_mark_btn']
                    d.click((x1 + x2) // 2, (y1 + y2) // 2)
                    if c > 0:
                        x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['delete_btn']
                        d.click((x1 + x2) // 2, (y1 + y2) // 2)

                        x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['firm_delete_btn']
                        if d.check_box_color([x1, y1, x2, y2], color='green', thrsh=0.1):
                            d.click((x1 + x2) // 2, (y1 + y2) // 2)
                        else:
                            click_on_x_icon(d)
                            print('======================done=====================')
                            print('stop')
                            break
                    x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['first_monster']
                    d.click((x1 + x2) // 2, (y1 + y2) // 2)
                    x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['central_box']
                    d.click((x1 + x2) // 2, (y1 + y2) // 2)

                    x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['attack_monster_btn']
                    d.click((x1 + x2) // 2, (y1 + y2) // 2)

                    x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['hint2']
                    monster_exists = d.check_hint([x1, y1, x2, y2], hint_monster_img)

                    if monster_exists:

                        x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['attack_btn_only']
                        if not d.check_box_color([x1, y1, x2, y2], color='green', thrsh=0.1):
                            x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['attack_btn']
                            if not d.check_box_color([x1, y1, x2, y2], color='green', thrsh=0.1):
                                x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['attack_btn2']
                                if not d.check_box_color([x1, y1, x2, y2], color='green', thrsh=0.1):
                                    c+=1
                                    x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['exit_btn']
                                    d.click((x1 + x2) // 2, (y1 + y2) // 2)
                                    continue
                        d.click((x1 + x2) // 2, (y1 + y2) // 2)

                        for troopid in ['troop1', 'troop2', 'troop4', ]:
                            x1, y1, x2, y2 = rally_coodinates_dict[screen_size][troopid]
                            d.click((x1 + x2) // 2, (y1 + y2) // 2)
                            # time.sleep(1)
                            # valid = d.check_general_valid(rally_coodinates_dict)

                            x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['general']
                            valid = not d.check_hint([x1, y1, x2, y2], hint_invalid_general_img,
                                                 thrsh_and=0.6, thrsh_xor=0.3, clear_noise=False, binary_thrsh=100)
                            if valid:
                                print('troopid', troopid)
                                break
                        else:
                            print('troops not valid')
                            click_on_x_icon(d)

                        x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['run_button']
                        d.click((x1 + x2) // 2, (y1 + y2) // 2)

                        x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['warning_firm_btn']
                        warning = d.check_box_color([x1, y1, x2, y2], color='green', thrsh=0.1)
                        if warning:
                            d.click((x1 + x2) // 2, (y1 + y2) // 2)
                            # x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['run_button']
                            # d.click((x1 + x2) // 2, (y1 + y2) // 2)
                        c += 1
                    else:
                        c+=1
                        x1, y1, x2, y2 = rally_coodinates_dict[screen_size]['wipe_none']
                        d.wipe(x1, y1, x2, y2)
                else:
                    print('troop_valid:', troop_valid)
                    time.sleep(30)

        if self.list_task[0]=='patrol':
            colors = self.list_task[3:] if len(self.list_task)>3 else ['whi', 'org']
            patrol_function(d, sleep=[int(self.list_task[1]), int(self.list_task[2])], colors=colors)
            click_on_x_icon(d)
        elif self.list_task[0]=='rally':
            mode = self.list_task[1] if len(self.list_task)>1 else 'troop'
            join_rally_function(d, mode=mode)
        elif self.list_task[0]=='wipehi':
            wipe_she(d, 15)
            say_hi(d, 10)
            click_on_x_icon(d)
        elif self.list_task[0]=='wipe':
            wipe_she(d, 15)
            click_on_x_icon(d)
        elif self.list_task[0]=='hi':
            say_hi(d, 10)
            click_on_x_icon(d)
        elif self.list_task[0]=='madi':
            thrsh = int(self.list_task[3]) if len(self.list_task) > 3 else 0.4
            cost_distance_y = int(self.list_task[2]) if len(self.list_task) > 2 else 200
            cost_distance_x = int(self.list_task[1]) if len(self.list_task) > 1 else 500
            find_mandi(d, cost_distance_x=cost_distance_x, cost_distance_y=cost_distance_y, thrsh=thrsh)

        elif self.list_task[0]=='monster':
            attact_monster(d)


        # click_on_x_icon(d)


def strew(thread_count=1, list_param=[]):
    # # GetDevices()
    # for m in [0,1]:
    for m in [0]:
        threading.Thread(target=main, args=(m,thread_count,list_param)).start()


def main(m, thread_count, list_param):

    device_serial = devices[m].serial
    print('device_serial:', device_serial)
    tk = ['nhinhlt@gmail.com|nhinh2202']
    time.sleep(10)
    # for i in range(m, len(tk), thread_count):
    #     mail = tk[i].strip()
    #     run = starts(m ,mail, device_serial)
    #     run.run()
    mail = tk[0].strip()
    run = starts(m ,mail, device_serial, list_param)
    run.run()
    print('==============================done===============================')

def get_screen():
    for i, device in enumerate(devices):
        image = devices[i].screencap()
        with open('{}_{}_screen.png'.format(i, devices[i].serial), 'wb') as f:
            f.write(image)
            print(os.path.abspath('{}_{}_screen.png'.format(i, devices[i].serial)))

if __name__ == '__main__':
    # main()
    if len(sys.argv) > 1:
        list_param = sys.argv[1:]
    strew(thread_count=1, list_param=list_param)
    # get_screen()
'''
python test_adb.py patrol 1 4
python test_adb.py madi 1000 200
763 320
774 323
785 327
'''