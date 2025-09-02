import cv2
import time
import mediapipe as mp
import os
import sujal_hand_track_module as stm
import numpy as np

brushthickness=12
eraserthickness=40


folderpath='vp imgs'
mylists=os.listdir(folderpath)
# print(mylists)
overlaylist=[]

for impath in mylists:
    image=cv2.imread(f'{folderpath}/{impath}')
    overlaylist.append(image)
# print(len(overlaylist))

header = overlaylist[0]
drawcolor=(0,0,255)


cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector=stm.handDetector(detectionCon=0.85)

xp,yp=0,0
imgcanvas=np.zeros((720,1280,3),np.uint8)


while True:
    suc,img=cap.read()
    img=cv2.flip(img,1)

    # setting the header image
    img[0:125,0:1280]=header 
    # img=cv2.addWeighted(img,0.5,imgcanvas,0.5,0)

    # find hand landmarks
    img=detector.findHands(img)
    lmlist,_ =detector.findposition(img,draw=False)
    if(len(lmlist)!=0):
        
        # tip of index and middle finger
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]

        # check which finger is up
        fingers=detector.fingersup()
        # print(fingers)

        # check if selection mode
        if fingers[1] and fingers[2]:
            xp,yp=0,0 
            # checking for the click and changing the image accordingly
            if y1<125:
                if 250<x1<450:
                    header = overlaylist[0]
                    drawcolor=(0,0,255)
                elif 550<x1<750:
                    header = overlaylist[1]
                    drawcolor=(0,255,0)
                elif 800<x1<950:
                    header = overlaylist[2]
                    drawcolor=(255,0,0)
                elif 1050<x1<1200:
                    header = overlaylist[3]
                    drawcolor=(0,0,0)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawcolor,cv2.FILLED)
            # print("Selection Mode")



        # check if drawing mode
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,drawcolor,cv2.FILLED)
            # print("Drawing Mode")

            if xp==0 and yp==0:
                xp , yp =x1,y1

            if drawcolor ==(0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawcolor,eraserthickness)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),drawcolor,eraserthickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawcolor,brushthickness)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),drawcolor,brushthickness)

            xp ,yp =x1 , y1
    imggray=cv2.cvtColor(imgcanvas,cv2.COLOR_BGR2GRAY)
    _, imginv=cv2.threshold(imggray,50,255,cv2.THRESH_BINARY_INV)
    imginv=cv2.cvtColor(imginv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imginv)
    img=cv2.bitwise_or(img,imgcanvas)

    #activating camera
    cv2.imshow("virtual pianter",img)
    # cv2.imshow("virtual canvas",imgcanvas)
    cv2.waitKey(1)