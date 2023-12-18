import os
#
import cv2
import numpy as np

def find_list(img_list=[], screen=None, list_name=[], threshold=0.9, f_name=''):

    print('screen', screen.shape)
    match = False
    for i, img in enumerate(img_list):
        screen_crop = screen
        result = cv2.matchTemplate(img, screen_crop, cv2.TM_CCOEFF_NORMED)
        # result = cv2.matchTemplate(img, screen_crop, cv2.TM_SQDIFF_NORMED) # # DIFF SIZE
        loc = np.where(result >= threshold)
        retVal = list(zip(*loc[::-1]))
        # # We want the minimum squared difference
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if retVal > [(0, 0)]:
            # print('retVal:', retVal)
            # image = cv2.rectangle(screen_crop, retVal[0],(retVal[0][0]+img.shape[0],retVal[0][1]+img.shape[1]), (0,250,0), 2)
            # if min_loc > (0,0):
            #     print('minloc:', min_loc, min_val)
            #     image = cv2.rectangle(screen_crop, min_loc,(min_loc[0]+img.shape[1],min_loc[1]+img.shape[0]), (0,0,255), 3)
            if max_loc > (0,0):
                ret = max_loc[0] + img.shape[1]//2, max_loc[1]+img.shape[0]//2
                # print('max_loc:', max_loc, max_val, ret)
                image = cv2.rectangle(screen_crop, max_loc,(max_loc[0]+img.shape[1],max_loc[1]+img.shape[0]), (0,0,255), 3)
                # name_box = [ret[0]-175,ret[1] + 100, ret[0]+200, ret[1] + 130]
                name_box = [ret[0]-175,ret[1] + 95, ret[0]-90, ret[1] + 130]
                image = cv2.rectangle(image, (name_box[0], name_box[1]),(name_box[2], name_box[3]), (0,0,255), 3)
                name = image_2_text(screen_crop[name_box[1]:name_box[3], name_box[0]:name_box[2] ], l='eng')
                print('======================================name:', name)

            cv2.imwrite('./debugs/9'+str(i)+ f_name + '_'+list_name[i], image)
            # print('./debugs/'+str(i)+ ' '+list_name[i])
            # match = True
            # # Naming a window
            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            #
            # cv2.resizeWindow("test", 1000, 2000)
            # cv2.imshow("test",image)
            # cv2.waitKey(0)
            # cv2.destroyWindow("test")
    return

def image_2_text(img, l='vie'):

    '''
    pip install pytesseract
    tesseract german.png stdout -l vie
    /home/nhinhlt/Downloads/tessdata-main
    export TESSDATA_PREFIX=/home/nhinhlt/Downloads/tessdata-main
    echo $TESSDATA_PREFIX
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
def find_list_debug():
    list_fns = [
        'mandi1.png',
        # 'mandi2.png',
        # 'mandi3.png',
        # 'mandi4.png',
        # 'add_mark.png',

                ]
    list_fns = [fn for fn in list_fns if os.path.exists('photos/{}'.format(fn))]
    print('list_fns', list_fns)
    img_list = [cv2.imread('photos/{}'.format(fn)) for fn in list_fns]
    screen_paths = [
                'photos/0_R58M706AGPL_screen_mandi1.png',
                'photos/0_R58M706AGPL_screen_mandi2.png',
                'photos/0_R58M706AGPL_screen_mandi3.png',
                'photos/0_R58M706AGPL_screen_mandi4.png',
                'photos/0_R58M706AGPL_screen_mandi5.png',
                'photos/0_R58M706AGPL_screen_mandi6.png',
                'photos/0_R58M706AGPL_screen_mandi7.png',
                'photos/0_R58M706AGPL_screen_mandi8.png',
                'photos/0_R58M706AGPL_screen_mandi10.png',
                'photos/0_R58M706AGPL_screen_mandi11.png',
                'photos/0_R58M706AGPL_screen_mandi12.png',
                'photos/0_R58M706AGPL_screen_mandi13.png',
                'Photos-002/0_glass.png',
                'Photos-002/1_glass.png',
                'Photos-002/9_trap3.png',
                ]
    for screen_path in screen_paths:
        screen_img = cv2.imread(screen_path)
        f_name = screen_path.split('/')[-1].replace('.png', '')
        print('f_name:', f_name)
        find_list( img_list=img_list, screen=screen_img, list_name=list_fns, f_name=f_name)

def find_list2(img_list=[], screen=None, list_name=[], threshold=0.99):
    screen_item_crop = screen[1110-0:1275+0, 440-0:630+0]
    screen_name_crop = screen[1295-0:1335+0, 110-0:955+0]
    # screen_item_crop = screen[1110-10:1275+10, 440-10:630+10]
    # screen_name_crop = screen[1295-10:1335+10, 110-10:955+10]

    print('screen', screen.shape)
    print('screen_item_crop', screen_item_crop.shape)
    print('screen_name_crop', screen_name_crop.shape)
    match = False
    for i, img in enumerate(img_list):
        # screen_crop = screen_item_crop if '1' in list_name[i] or '2' in list_name[i] or '3' in list_name[i]  else screen_name_crop
        screen_crop = screen_item_crop if img.shape[1]/img.shape[0] < 2 else screen_name_crop
        # screen_crop = screen
        scale = min(screen_crop.shape[1] / img.shape[1] , screen_crop.shape[0] / img.shape[0])
        if scale < 1:
            # print(list_name[i], img.shape, scale)
            img = cv2.resize(img, None, fy=scale, fx=scale)
        result = cv2.matchTemplate(img, screen_crop, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        retVal = list(zip(*loc[::-1]))

        if retVal > [(0, 0)]:
            image = cv2.rectangle(screen_crop, retVal[0],(retVal[0][0]+img.shape[0],retVal[0][1]+img.shape[1]), (0,250,0), 2)
            cv2.imwrite('./debug/'+str(i)+ '_'+list_name[i], image)
            print('./debug/'+str(i)+ ' '+list_name[i])
            match = False
            # cv2.imshow("test",image)
            # cv2.waitKey(0)
            # cv2.destroyWindow("test")
    return match

def find_list_debug2():
    '''
    pip install pytesseract
    tesseract german.png stdout -l vie
    /home/nhinhlt/Downloads/tessdata-main
    export TESSDATA_PREFIX=/home/nhinhlt/Downloads/tessdata-main
    echo $TESSDATA_PREFIX
    :return:
    '''
    import pytesseract
    config = ("-l vie")
    # config = ("-l eng --oem 1 --psm 7")
    # text = pytesseract.image_to_string(roi, config=config)
    # screen_path = 'photos/Screenshot_20231214-102108_Evony.jpg' # 10k stone
    # screen_path = 'photos/Screenshot_20231214-102141_Evony.jpg' # wheat 1
    # screen_path = 'photos/Screenshot_20231214-103150_Evony.jpg' # forg 1
    # screen_path = 'photos/Screenshot_20231214-103221_Evony.jpg' # forg 3
    # screen_path = 'photos/Screenshot_20231214-103232_Evony.jpg' # trap 3
    # screen_path = 'photos/Screenshot_20231214-103256_Evony.jpg' # crow train 1
    # screen_img = cv2.imread(screen_path)
    # screen_name_crop = screen_img[1295-0:1335+0, 110-0:955+0]
    # text = pytesseract.image_to_string(screen_name_crop, config=config)
    # text = pytesseract.image_to_string(screen_name_crop)
    item_name = 'Phân thưởng Tuân tra:Vương miếRm Bên bỉíX'
    item_name_short =  item_name[item_name.find('tra:') + 4:]
    print('text:', item_name_short)
def check_green(img):

    hsvFrame  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    red_lower = np.array([0, 0, 0], np.uint8) #red 0 -10
    red_upper = np.array([10, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    green_lower = np.array([50, 0, 0], np.uint8) # green 50 -60
    green_upper = np.array([60, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    if np.sum(green_mask) > np.sum(red_mask):
        return True
    return False
def improve_general_check(screen, require=['leader', 'attack', 'defense', 'politics']):
    '''

    :return:
    '''

    leader_crop = screen[461:495, 925:1010]
    attack_crop = screen[540:574, 925:1010]
    defense_crop = screen[620:652, 925:1010]
    politics_crop = screen[698:732, 925:1010]

    cv2.imwrite('crop1.png', leader_crop)
    cv2.imwrite('crop2.png', attack_crop)
    cv2.imwrite('crop3.png', defense_crop)
    cv2.imwrite('crop4.png', politics_crop)

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
    if confirm:
        print('confirm add', leader_add,  attack_add, defense_add, politics_add,)
    else:
        print('skip add', leader_add,  attack_add, defense_add, politics_add,)
    return confirm

def nemove_light_color():
    # import numpy as np
    # Load image
    imp = '/home/nhinhlt/game/python_adb/debugs/black5.png'
    # imp = '/home/nhinhlt/game/python_adb/debugs/1_imgs_remove_0_255_0.png'
    # imp = '/home/nhinhlt/game/python_adb/debugs/gray.png'
    # imp = '/home/nhinhlt/game/python_adb/debugs/screen_rally1.png'
    im = cv2.imread(imp)
    hsvFrame  = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)

    # red_lower = np.array([50, 80, 0], np.uint8) #green
    # red_upper = np.array([60, 255, 150], np.uint8)
    red_lower = np.array([0, 150, 0], np.uint8) # black v 0-50, gray v 0 - 155
    red_upper = np.array([30, 255, 150], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    im[red_mask>0] = [255,255,255]
    cv2.imwrite(imp.replace('.png' , 'imgs_remove_black.png'), im)
    if np.sum(red_mask) > red_mask.shape[0] * red_mask.shape[1] * 0.5 * 255:
        print('red mask')
        return True

    # if np.sum(red_mask) > np.sum(green_mask):
    #     im[red_mask>0] = [255,255,255]
    #     cv2.imwrite(imp.replace('.png' , 'imgs_remove_red.png'), im)
    # else:
    #     im[green_mask>0] = [255,255,255]
    #     cv2.imwrite(imp.replace('.png' , 'imgs_remove_green.png'), im)


    # # Define lower and upper limits of our blue
    # RMin = np.array([0, 100, 50],np.uint8) # màu độ, bão hòa , độ sáng
    # RMax = np.array([200, 255, 255],np.uint8)

    # Go to HSV colourspace and get mask of blue pixels
    # HSV  = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(HSV, RMin, RMax)

    # Make all pixels in mask white
    # im[mask>0] = [255,255,255]
    # im = cv2.bitwise_and(im, im, mask=mask)
    # cv2.imwrite(imp.replace('.jpg' , 'imgs_remove_red.png'), im)

def gen_debug():
    list_fns = [
            'Photos-001/Screenshot_20231215-153405_Evony.jpg',
            'Photos-001/Screenshot_20231215-153415_Evony.jpg',
            'Photos-001/Screenshot_20231215-153430_Evony.jpg',
            'Photos-001/Screenshot_20231215-153452_Evony.jpg',
            'Photos-001/Screenshot_20231215-153508_Evony.jpg',
            'Photos-001/Screenshot_20231215-153535_Evony.jpg',
            'Photos-001/Screenshot_20231215-153559_Evony.jpg',
            'Photos-001/Screenshot_20231215-153617_Evony.jpg',
                ]
    for screen_path in list_fns:
        print(screen_path)
        screen = cv2.imread(screen_path)
        firm = improve_general_check(screen, require=['defense'])
        # firm = improve_general_check(screen, require=['politics', 'attack', 'defense', 'leader'])
        if firm:
            print('firm:============================================', )

def patrol_check_color(screen):
    '''
    :return:
    '''

    def check_orange(img):

        hsvFrame  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        org_lower = np.array([0, 0, 0], np.uint8) #orange 0 -20
        org_upper = np.array([20, 255, 255], np.uint8)
        org_mask = cv2.inRange(hsvFrame, org_lower, org_upper)

        if np.sum(org_mask) > org_mask.shape[0]*org_mask.shape[1]*0.7*255:
            return True
        return False

    def check_white(img):

        hsvFrame  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        org_lower = np.array([0, 0, 200], np.uint8) #orange
        org_upper = np.array([255, 255, 255], np.uint8)
        org_mask = cv2.inRange(hsvFrame, org_lower, org_upper)

        # Make all pixels in mask white
        img[org_mask>0] = [255,255,255]
        img = cv2.bitwise_and(img, img, mask=org_mask)
        cv2.imwrite('imgs_remove_white.png', img)
        if np.sum(org_mask) > org_mask.shape[0]*org_mask.shape[1]*0.3*255:
            return True
        return False

    if screen.shape[0] > 2000:
        screen_item_edge_crop = screen[1120:1140, 480:483]
    else:
        screen_item_edge_crop = screen[741:755, 320:323]

    cv2.imwrite('screen_item_edge_crop_white.png', screen_item_edge_crop)
    confirm = check_white(screen_item_edge_crop)
    # confirm = check_orange(screen_item_edge_crop)

    if confirm:
        print('confirm white' )
        # print('confirm orange' )
    return confirm

def patrol_color_debug():
    list_fns = [
        # '/home/nhinhlt/game/python_adb/Photos-002/1_iron3.png',
        # '/home/nhinhlt/game/python_adb/Photos-002/2_trap3.png',
        # '/home/nhinhlt/game/python_adb/Photos-002/3_glass.png',
        # '/home/nhinhlt/game/python_adb/Photos-002/3_trap3.png',
        # '/home/nhinhlt/game/python_adb/Photos-002/screen.png',
        '/home/nhinhlt/game/python_adb/Photos-002/screen2.png',
                ]
    for screen_path in list_fns:
        print(screen_path)
        screen = cv2.imread(screen_path)
        firm = patrol_check_color(screen)
        # firm = (screen, require=['politics', 'attack', 'defense', 'leader'])
        if firm:
            print('firm:============================================', )

def rally_curr(screen):
    '''
    :return:
    '''

    '''
    :return:
    '''

    def check_color(img, lower_h=0, upper_h=10, lower_v=0, upper_v=255,
                    lower_s=0, upper_s=255, thrsh=0.5, debug=True, count=0):
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
            cv2.imwrite('debugs/{}_imgs_remove_{}_{}_{}.png'.format(count, lower_h, upper_h, lower_v), img)
            count += 1
        if np.sum(mask) > mask.shape[0] * mask.shape[1] * thrsh * 255:
            return True
        return False
    rally_coodinates_dict = {2000: {'screen_rally1': [806, 992, 1024, 1058], 'rally_bock_h': 636,
                                    'run_button':[575, 2219, 971, 2302], 'reset_botton':[120, 2219, 482, 2302],
                                    'add_1_best':[734, 1759, 753, 1782],
                                    'troop1':[53, 250, 162, 350],
                                    'troop2':[183, 250, 285, 350],
                                    'troop3':[306, 250, 400, 350],
                                    'troop4':[429, 250, 526, 350],
                                    'general':[74, 1054, 276, 1252],
                                    },
                             1600: {'screen_rally1': [806, 992, 1024, 1058], 'rally_bock_h': 230,
                                    'run_button':[575, 2219, 971, 2302],'reset_botton':[120, 2219, 482, 2302]}
                             }
    if screen.shape[0] > 2000:
        x1, y1, x2, y2 = rally_coodinates_dict[2000]['screen_rally1']
        rally_bock_h = rally_coodinates_dict[2000]['rally_bock_h']
    else:
        x1, y1, x2, y2 = rally_coodinates_dict[1600]['screen_rally1']
        rally_bock_h = rally_coodinates_dict[1600]['rally_bock_h']

    screen_rally1 = screen[y1:y2, x1:x2]
    screen_rally2 = screen[y1 + rally_bock_h:y2 + rally_bock_h, x1:x2] # 1623 - 982
    screen_rally3 = screen[y1 + rally_bock_h*2:y2 + rally_bock_h*2, x1:x2]

    # cv2.imwrite('debugs/screen_rally1.png', screen_rally1)
    # cv2.imwrite('debugs/screen_rally2.png', screen_rally2)
    # cv2.imwrite('debugs/screen_rally3.png', screen_rally3)
    confirm = []

    rally_time_crop3 = screen_rally3[screen_rally3.shape[0] // 2:].copy()
    rally3 = check_color(screen_rally3, lower_h=50, upper_h=60, thrsh=0.7, lower_s=80, upper_s=255, lower_v=0, upper_v=150)
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
    rally1 = check_color(screen_rally1, lower_h=50, upper_h=60, thrsh=0.7, lower_s=80, upper_s=255, lower_v=0, upper_v=150)
    rally_time1 = check_color(rally_time_crop1,
                              lower_h=0, upper_h=10, thrsh=.1)
    if rally1 and not rally_time1:
        confirm.append(1)
        print('confirm rally1', confirm)
        return confirm[0]
    return 0

def rally_curr_debug():
    list_fns = [
        '/home/nhinhlt/game/python_adb/photos/0_R58M706AGPL_screen.png',
        '/home/nhinhlt/game/python_adb/photos/0_R58M706AGPL_screen2.png',
        # '/home/nhinhlt/game/python_adb/photos/0_R58M706AGPL_screen3.png',
        '/home/nhinhlt/game/python_adb/photos/0_R58M706AGPL_screen4_debug.png',
                ]
    for screen_path in list_fns:
        print(screen_path)
        screen = cv2.imread(screen_path)
        curr_id = rally_curr(screen)
        if curr_id:
            print('rally:============================================', curr_id)


def patrol_check():
    '''
    :return:
    '''
    from editdistance import distance
    list_text = [
        'Phân thưởng Tuần tra: Vương miện-#hfếLquân X.',
        'Phân thưởng: Tuân tra: Hôp49K Đá X 1',
        'Phân thưởng Luân tra: Pha lêAñH sáng X1',
        'Phần thưởng Tuân tra: Vương miên##rfh-.hoat X 1',
        'Phần thưởng Tuân tra: Vương miên##rfh-.hoat X 1',
        'Phân thưởng Tuân tra:Vương miếRm Bên bỉíX',
        'Phân thưởng Tuần tra: Vương miện-#hfếLquân X.',

        'Phân-hưởng Tuân ra: PhaslGStó X 1',
        'Phân thưởngcLóuân tra: 24 GiờzZEãnG:tốc X 11',
        'Phân thưởng Tuân:tra: Pha.l&t ửa X 1',
        'Phân thưởng Tuân tra: 30 PhútTAnñg tốc X1',
        'Phân thưởng Tuân tra::-Huân chướ8G:Sấm X1',
    ]
    for item_name in list_text:
        item_name_short = item_name[item_name.find('tra:') + 4:].strip().replace(':', '').replace('-', '').replace('.', '')
        if 'Tuân' in item_name_short or 'thưởng' in item_name_short:
            item_name_short = item_name[item_name.find('ra:') + 3:].strip().replace(':', '').replace('-', '').replace('.', '')
        list_key = ['Vương miện', 'Pha lê', 'Sừng', 'Quyền trượng', 'Chén thánh', 'Huân chương',
                    '24 Giờ',
                    'Phal&t', 'PhaslG',
                    ]
        for k in list_key:
            if k in item_name:
                # print('=========ok======', item_name, item_name_short[:len(k)], k)
                break
            elif distance(item_name_short[:len(k)], k) < len(k)/3:
                # print('=========ok======', item_name, item_name_short[:len(k)], k)
                break
            else:
                pass
                # print('none', item_name_short[:len(k)], k, distance(item_name_short[:len(k)], k), len(k)/3)
        else:
            print('skip', item_name, item_name_short)
    return False

def mode_color():
    # importing required packages
    from scipy import stats as st
    import numpy as np

    # creating an array using array() method
    abc = np.array([1, 1, 2, 2, 2, 3, 4, 5])

    # applying mode operation on array and
    # printing result
    print(st.mode(abc))

def mode_color2():
    # Program to find most frequent
    # element in a list

    from collections import Counter

    def most_frequent(List):
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0]

    List = [2, 1, 2, 2, 1, 3]
    print(most_frequent(List))

def tmp():
    screen_size = 2000
    rally_coodinates_dict = {'2000': {
        'wipe_up': [964, 600, 104, 1400],
        'wipe_down': [104, 1400, 964, 600],
        'wipe_left': [104, 600, 964, 1400],
        'wipe_right': [964, 1400, 104, 600],
        'add_botton': [295, 1476, 766, 1558],
    }
    }
    print(rally_coodinates_dict[str(screen_size)]['add_botton'])

if __name__ == '__main__':
    tmp()
    # find_list_debug()
    # find_list_debug2()
    # patrol_check()
    # gen_debug()
    # patrol_color_debug()
    # rally_curr_debug()
    # nemove_light_color()
    # strew()
# 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
###  [(405, 1096), (406, 1096), (405, 1097), (406, 1097), (405, 1098), (406, 1098), (405, 1099), (406, 1099)]