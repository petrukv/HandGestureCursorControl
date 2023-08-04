import cv2 as cv 
import mediapipe as mp
import numpy as np
import pyautogui as pg 

cap = cv.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

screenWidth, screenHeight = pg.size()
print(screenWidth, screenHeight)

frameR = 100

tipid = [4, 8, 12, 16, 20]

clk = 1

while True:
    succes, img = cap.read()
    img = cv.flip(img, 1)

    h, w, c = img.shape

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    cv.rectangle(img, (frameR, frameR), (w-frameR, h-frameR), (255,0,0), 2)

    if (results.multi_hand_landmarks):

        tangan = results.multi_handedness[0].classification[0].label
        if tangan == "Right":
            lmlist = []

            for handlms in results.multi_hand_landmarks:
                for id, landmarks in enumerate(handlms.landmark):
                    cx, cy = int(landmarks.x*w), int(landmarks.y*h)
                    lmlist.append([id, cx, cy])

            fingers = []

            if lmlist[tipid[0]][1] < lmlist[tipid[0]-2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if lmlist[tipid[id]][2] < lmlist[tipid[id]-3][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            if fingers == [0, 1, 1, 0, 0]:
                cv.circle(img, (lmlist[12][1], lmlist[12][2]), 10, (0, 0, 225), cv.FILLED)

                X = np.interp(lmlist[12][1], (frameR, w-frameR), (0, screenWidth))
                Y = np.interp(lmlist[12][2], (frameR, h-frameR), (0, screenHeight))

                length = abs(lmlist[8][1] - lmlist[12][1])

                if length > 50:
                    pg.click()
                    clk = -1
                
                if length <20:
                    pg.moveTo(X, Y, duration=0.1)

            if fingers == [1, 1, 0, 0, 0]:
                cv.circle(img, (lmlist[4][1], lmlist[4][2]), 10, (0,0,255), cv.FILLED)
            
                X = np.interp(lmlist[4][1], (frameR, w-frameR), (0, screenWidth))
                Y = np.interp(lmlist[4][2], (frameR, h-frameR), (0, screenHeight))

                length = abs(lmlist[4][1] - lmlist[8][1])

                if length > 50:
                    pg.scroll(100)

                else:
                    pg.scroll(-100)

                
    cv.imshow("webcam", img)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

cv.destroyAllWindows()
