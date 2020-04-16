from datetime import date
import sys
import os
import cv2

haar_file = 'haarcascade_frontalface_default.xml'


try:
    name = sys.argv[1]
except IndexError as error:
    print("YOU NEED TO PASS ARGUMENT");
    sys.exit();

datasets = 'datasets'
today = date.today()
today_converted = date.today().strftime('%d-%m-%Y')


path = os.path.join(datasets, name, today_converted)
if not os.path.isdir(path):
    os.makedirs(path)
(width, height) = (130, 100)  # defining the size of image

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)  # '0' is use for my webcam, if you've any other camera attached use '1' like this

# The program loops until it has 30 images of the face.
count = 1
while count < 30:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
    count += 1

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break
