import numpy as np 
from PIL import Image
from os import listdir
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt


def extract(file):
    img=Image.open(file)
    img=img.convert('RGB')
    img_array=np.asarray(img)
    face_detector=MTCNN()
    pixels=face_detector.detect_faces(img_array)
    x1,y1,width,height=pixels[0]['box']
    
    x1,y1=abs(x1),abs(y1)
    x2=x1+width
    y2=y1+height

    face_pixels=img_array[y1:y2,x1:x2]
    face_image=Image.fromarray(face_pixels)
    face_image=face_image.resize((128,128))
    face_array=np.asarray(face_image)
    return face_array


folder='/home/clown/test_images/'
for file in listdir(folder):
    path=folder+file
    face=extract(path)
    plt.axis('off')

    plt.imshow(face)

plt.show()