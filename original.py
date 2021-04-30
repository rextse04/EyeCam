import cv2
import numpy as np
from tensorflow.keras.models import load_model
from getpass import getpass
# Load cascade
face_cascade = cv2.CascadeClassifier("face_cascade.xml")
eye_cascade = cv2.CascadeClassifier("eye_cascade.xml")
# Load model
model = load_model("generator.h5")
# Name window and bind click event
cv2.namedWindow("Stream")
def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        for (ex,ey,ew,eh) in eyes:
            if ex <= x - fx <= ex + ew and ey <= y - fy <= ey + eh:
                input_img = cv2.resize(roi_color[ey:ey + eh,ex:ex + ew],(32,32)) / 255
                output_img = (model.predict(np.expand_dims(input_img,0))[0] * 255).astype("uint8")
                cv2.imshow("Model prediction",output_img)
cv2.setMouseCallback("Stream",click_event)
# Get credentials
"""url = input("IP adress of the source: ")
username = input("Username: ")
password = getpass()"""
url = input("IP Address: ")
username = input("Username: ")
password = input("Password: ")
cap = cv2.VideoCapture("rtsp:" + username + ":" + password + "@" + url)
# Main application
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (fx,fy,fw,fh) in faces:
        cv2.rectangle(frame,(fx,fy),(fx + fw,fy + fh),(255,0,0),2)
        roi_gray = gray[fy:fy + fh,fx:fx + fw]
        roi_color = frame[fy:fy + fh,fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex + ew,ey + eh),(0,255,0),2)
    cv2.imshow("Stream",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()