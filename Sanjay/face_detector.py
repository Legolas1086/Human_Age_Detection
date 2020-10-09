import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
from os import listdir


def detect(file,label):
    #reading the image
    img=Image.open(file)
    img=img.convert('RGB')

    #converting image to numpy array
    img_array=np.asarray(img)


    face_detector=MTCNN()

    #detecting face using MTCNN object
    face_pixels=face_detector.detect_faces(img_array)

    #getting location of faces,eyes in the image
    for i in face_pixels:
        x1,y1,width,height=i['box']

        #drawing rectangle around the face
        cv2.rectangle(img_array,(x1,y1),(x1+width,y1+height),(0,255,0),2)
        cv2.putText(img_array,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0),2)

    return img_array    

