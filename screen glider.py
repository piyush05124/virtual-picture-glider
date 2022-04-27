#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import numpy as np


# In[2]:


import keyboard as key
import time


# In[26]:


orig=[]
xlast=[]
cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  
        results = hands.process(image)
        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for num,hand_s in enumerate(results.multi_hand_landmarks):
                
                #print(tuple(  (X,y)  ))
                mp_drawing.draw_landmarks(image, hand_s, mp_hands.HAND_CONNECTIONS)
            x,y=int(hand_s.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width),int(hand_s.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)  
            if results:
                orig.append([x,y])   # first occurance
                xlast.append(x-orig[0][0])
                #print(xlast[-1])
                if xlast[-1]<-70:
                    time.sleep(0.5)
                    key.press_and_release('left')
                    
                if xlast[-1]>70:
                    time.sleep(0.5)
                    key.press('right')
            else:
                orig.clear()
                xlast.clear()
                
                    
            
        else:
            orig.clear()
            xlast.clear()
                
            
            
  
            
        cv2.imshow(' Hands mouse', image)
        #print(xlast)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()          


# In[ ]:




