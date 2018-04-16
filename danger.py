import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
danger = cv2.imread('C:/Users/Anurag Shukla/Desktop/danger.jpg')
cap = cv2.VideoCapture(0)
k = 0
p = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = img[y:y+h, x:x+h]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0),1)
        if len(eyes) > 0:
            p = 1
        if p == 1:
            if len(eyes) == 0:
                k += 1
            
        if k == 30:
            cv2.imshow('danger',danger)
            os.startfile(r'C:\Users\Anurag Shukla\Downloads\danger_audio.mp3')
        print(eyes)
    
    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('1'):
        break
cap.release()
cv2.destroyAllWindows()
