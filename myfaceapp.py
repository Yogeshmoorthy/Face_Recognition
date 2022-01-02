import cv2
import numpy as np
import training

people=training.people
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

features=np.load('features.npy',allow_pickle=True)
labels=np.load('labels.npy',allow_pickle=True)

face_recog=cv2.face.LBPHFaceRecognizer_create()
face_recog.read('face_trained.yml')

# img=cv2.imread(r'C:\Users\yoges\.PyCharmCE2019.2\config\scratches\images\yogesh\yogesh3.jpg')
# cv2.imshow('ORIGINAL',img)
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("GRAY",gray)

from_cam=cv2.VideoCapture(0)
while (from_cam.isOpened()):
    ret,frame=from_cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detect=classifier.detectMultiScale(gray,1.5,3)
    print('No. of faces detected is:',len(face_detect))

    for (x,y,w,h) in face_detect:
        roi=gray[y:y+h, x:x+h]
        lab,conf=face_recog.predict(roi)
        # print("Label",lab)
        # print("Confidence",conf)
        # print("Person: ",people[lab])
        cv2.putText(frame,str(people[lab]), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0),thickness=2)
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), thickness=2)


    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imshow("Detected Result",frame)

# from_cam.release()
# cv2.imshow("Detected",frame)

# cv2.waitKey(0)
# cv2.destroyAllWindows()





















