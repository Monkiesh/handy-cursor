import cv2
import mediapipe as mp
import sys
from pynput.mouse import Controller, Button
import tkinter

screen_width = tkinter.Tk().winfo_screenwidth()  # 计算机屏幕水平分辨率
screen_height = tkinter.Tk().winfo_screenheight()  # 计算机屏幕垂直分辨率

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

mouse = Controller()
thumb_pos_x = [0.5, 0.5, 0.5, 0.5, 0.5] #防抖
thumb_pos_y = [0.5, 0.5, 0.5, 0.5, 0.5]
i_thumb = 0
is_click = True

def map_pos(x, y):
    ans_x = 0.5
    ans_y = 0.5
    if (x < 0.3):
        ans_x = 0
    if (y < 0.3):
        ans_y = 0
    if (x > 0.7):
        ans_x = 1
    if (y > 0.7):
        ans_y = 1
    if (ans_x == 0.5):
        ans_x = 2.5 * (x-0.3)
    if (ans_y == 0.5):
        ans_y = 2.5 * (y-0.3)
    return ans_x * screen_width, ans_y * screen_height

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)
            finger_fold_status = []
            for tip in finger_tips:
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                # print(id, ":", x, y)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            #print(finger_fold_status)

            if all(finger_fold_status):
                # like
                if lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y < lm_list[thumb_tip - 2].y:
                    #mouse.move((lm_list[5].x - thumb_pos_x) * 5000, (lm_list[5].y - thumb_pos_y) * 5000)
                    
                    if (i_thumb > 4):   #循环数组
                        i_thumb = 0
                    thumb_pos_x[i_thumb] = lm_list[5].x
                    thumb_pos_y[i_thumb] = lm_list[5].y

                    mouse.position = map_pos(sum(thumb_pos_x)/5, sum(thumb_pos_y)/5)
                    i_thumb = i_thumb + 1
                    #print(f"LIKE {mouse.position}")
                    
                    # Click
                    if(abs(lm_list[thumb_tip].y - lm_list[6].y) < abs(lm_list[10].y - lm_list[6].y) and is_click):
                        mouse.click(Button.left, 1)
                        is_click = False
                    else:
                        is_click = True
                        
                    
                # Dislike
                elif lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y > lm_list[thumb_tip - 2].y:
                    print("DISLIKE")
                    sys.exit()

                
                    
            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2)
                                   )

    cv2.imshow("Hand Sign Detection", img)
    cv2.waitKey(1)
