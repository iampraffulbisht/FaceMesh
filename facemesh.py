import cv2 as cv
import mediapipe as mp
import time



# cap = cv.VideoCapture("FaceMesh/video/2.MOV")
cap = cv.VideoCapture(0)
pTime=0


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=2)
while True:
    success,img = cap.read()
    imgRGB= cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mp.solutions.face_mesh_connections.FACEMESH_TESSELATION,drawSpec,drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                # print(lm)
                ih,iw,ic= img.shape
                x,y=int(lm.x*ih),int(lm.y*ih)
                print(id,x,y)






    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv.putText(img, f'FPS:{int(fps)}',(20,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv.imshow("Video",img)
    cv.waitKey(10)
