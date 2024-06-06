import cv2 as cv
import numpy as np 
import mediapipe as mp 
import time 

class FaceMeshDetector():
    def __init__(self,staticMode = False, maxFaces = 2,refine_landmarks=True, minDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refine_landmarks= refine_landmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.refine_landmarks,self.minDetectionCon,self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=3,circle_radius=2,color=(144,238,144))
        self.landmark_drawing_spec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

    def findFaceMesh(self,img,draw=True):
        self.imgRGB= cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces=[]
        
        if self.results.multi_face_landmarks:
                for faceLms in self.results.multi_face_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img,faceLms,mp.solutions.face_mesh_connections.FACEMESH_TESSELATION,self.landmark_drawing_spec,self.drawSpec) #mp.solutions.face_mesh_connections.FACEMESH_TESSELATION self.drawSpec
                        face = []
                        for id,lm in enumerate(faceLms.landmark):
                            # print(lm)
                            ih,iw,ic= img.shape
                            x,y=int(lm.x*iw),int(lm.y*ih)
                            # cv.putText(img, str(id),(x,y),cv.FONT_HERSHEY_PLAIN,0.5,(0,255,0),1)
                            # print(id,x,y)
                            face.append([x,y])
                        faces.append(face)
        return img,faces

def main():
    # cap = cv.VideoCapture('FaceMesh/video/1.mp4')
    cap = cv.VideoCapture(0)
    pTime=0
    detector = FaceMeshDetector()

    while True:
        success,img = cap.read()
        img,faces = detector.findFaceMesh(img,True)

        if len(faces)!=0:
            print(len(faces[0]))
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime
        cv.putText(img, f'FPS:{int(fps)}',(20,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv.imshow("Video",img)
        cv.waitKey(10)




if __name__ == "__main__":
    main()