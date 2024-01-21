import numpy as np
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap =cv2.VideoCapture(0)
detector= FaceMeshDetector(maxFaces=1)
textList =["Welcome to my world",
           "You are trapped ",
           "i will kill you",
           "Alright Alright Alright", "Whose next"
           ,"I am bored"]
while True:
    success, img= cap.read()
    imageText = np.zeros_like(img)
    img , faces =detector.findFaceMesh(img,draw=False)


    if faces:
        face = faces[0]
        pointleft =face[145]
        pointright =face[374]


        # cv2.line(img, pointleft,pointright,(0,200,0),3)
        # cv2.circle(img,pointleft,5,(255,0,255),cv2.FILLED)
        # cv2.circle(img,pointright,5,(255,0,255),cv2.FILLED)
        w,_ = detector.findDistance(pointleft,pointright)
        W=6.3
        w=w/100
        f=10.2
        d=(W*f)/w
        i=0
        

        cvzone.putTextRect(img,f'Depth :{int(d)}cm',
                           (face[10][0]-200,face[10][1]-50),scale=2)
        for  text in textList:
            
            singleHeight =50 +int(d/2)
            scale=0.4+ int(d/10)*10/40
            cv2.putText(imageText,text,(50,50+(i*singleHeight)),
                         cv2.FONT_ITALIC, scale ,(255,255,255),2)
            i+=1



    imgStacked = cvzone.stackImages([img,imageText],2,1)
    cv2.imshow("Image",imgStacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
