#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import numpy as np


# In[9]:


import mouse
import time


# results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

# In[19]:


##htrack=r"Z:\btest.jpg"
##image=cv2.imread(htrack)


# In[20]:


##import cv2
##import mediapipe as mp
##mp_drawing = mp.solutions.drawing_utils
##mp_hands = mp.solutions.hands
##li=[]
### For static images:
##with mp_hands.Hands(
##    static_image_mode=True,
##    max_num_hands=2,
##    min_detection_confidence=0.5) as hands:
##    image = cv2.flip(image,1)#cv2.imread(htrack), 1)
##
##    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
##    image_height, image_width, _ = image.shape
##    annotated_image = image.copy()
##    for i,hands in enumerate(results.multi_hand_landmarks):
##        print('Index finger tip coordinates for {}: ('.format(i),int(hands.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width), int(hands.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height),')')
##        mp_drawing.draw_landmarks(annotated_image, hands, mp_hands.HAND_CONNECTIONS)
##    for j,cls in enumerate(results.multi_handedness):
##        li.append(cls.classification[0].label)
##    print(li)
##    cv2.imwrite('Z:/annotated_image_1'+ '.png', cv2.flip(annotated_image, 1))
##

# In[ ]:





# In[22]:




font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 2
color = (255, 0, 0)
thickness = 2


lh=[]
rh=[]

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.4) as hands:
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
            for idx,cls in enumerate(results.multi_handedness) :
                
                x,y=int(hand_s.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width),int(hand_s.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                a,b=int(hand_s.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width),int(hand_s.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
                cv2.circle(image,(x,y), 10, (0,100,255), 5)
                cv2.circle(image,(a,b), 10, (100,100,255), 5)
                time.sleep(0.1)
                mouse.move(x ,y)
                dist=np.sqrt( (x-a)**2+(y-b)**2 )
                if dist<20:
                    time.sleep(0.1)
                    mouse.click('left')
                cv2.line(image,(x,y),(a,b),(255,255,255),3)
                cv2.putText(image,str(int(round(dist,2))), org, font,fontScale, color, thickness, cv2.LINE_AA)
                
                
                
                
                
                
                
        cv2.imshow(' Hands mouse', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()                

