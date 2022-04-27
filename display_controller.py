#!/usr/bin/env python
# coding: utf-8

# In[2]:

import screen_brightness_control as sbc
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# In[3]:


import numpy as np


# In[5]:


# For webcam input:
def get_coord(po):
    nn=np.array((po.x,po.y))
    kk=tuple(np.multiply(nn,[640,480]).astype(int))
    return kk[0],kk[1]


font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 2
color = (255, 0, 0)
thickness = 2


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_handedness:
            for num,hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
            for idx,clss in enumerate(results.multi_handedness):
                ind=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                th=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
                x,y=get_coord(ind)
                X,Y=get_coord(th)
                cv2.circle(image,(x,y), 9, (0,255,255), -1)
                cv2.circle(image,(X,Y), 9, (0,100,255), 5)
                cv2.line(image, (X,Y),(x,y), (123,43,240), 3)
                cv2.circle(image,((X+x)//2,(Y+y)//2), 9, (122,255,255), -1)
                dist=int(round(np.sqrt((X-x)**2+(Y-y)**2)))
                try:
                    sbc.set_brightness(dist)
                except Exception:
                    cv2.putText(image,'brigtness value exceeded', (50,100), font,fontScale, color, thickness, cv2.LINE_AA)
                    
                cv2.putText(image,str(int(round(np.sqrt((X-x)**2+(Y-y)**2),2))), org, font,fontScale, color, thickness, cv2.LINE_AA)
                    #print(np.sqrt((X-x)**2+(Y-y)**2))
                
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()

