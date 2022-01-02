import os
import cv2

img_path=(r'C:\Users\yoges\.PyCharmCE2019.2\config\scratches\images')
folder_name=str(input('Enter the name: '))
folder_path=(''.join([img_path,'\\',folder_name]))
print(folder_path)
for dirpath,dirname,filename in os.walk(img_path):
    try:
        if folder_path != dirpath:
            os.mkdir(folder_path)
            break
    except FileExistsError as e:
        print('Already is there')

video=cv2.VideoCapture(0)
while True:
    ret,frame=video.read()
    cv2.imshow("My window",frame)
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray window',gray_img)
    for i in range(100):
        os.chdir(folder_path)
        cv2.imwrite(filename=''.join([folder_name,str(i),'.jpg']),img=frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()






