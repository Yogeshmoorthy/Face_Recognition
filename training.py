import cv2
import pandas as pd
import numpy as np
import os

dir=(r'C:\Users\yoges\.PyCharmCE2019.2\config\scratches\images')
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

people=[]
features=[]
labels=[]
for dirpath,dirname,filename in os.walk(dir):
    for direc in dirname:
        people.append(direc)

def create_train():
    for person in people:
        path=os.path.join(dir,person)
        label=people.index(person)
        for i in os.listdir(path):
            img_path=os.path.join(path,i)
            infolder=cv2.imread(img_path)
            if infolder is None:
                continue

            gray_img=cv2.cvtColor(infolder,cv2.COLOR_BGR2GRAY)
            cascade=classifier.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=4)

            for (x,y,w,h) in cascade:
                roi=gray_img[y:y+h, x:x+w]
                features.append(roi)
                labels.append(label)

create_train()

features=np.array(features,dtype='object')
labels=np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)








