from PIL import Image
import matplotlib.pyplot as plt
from face_extraction import extract
from face_detector import detect
import numpy as np 
from tensorflow.keras.models import load_model
from os import listdir
import cv2
import pandas as pd

model=load_model('/home/clown/age_detection.h5')
data=pd.read_csv('/home/clown/VS/Age_detection/age_gender.csv',delimiter=',')
y=data.iloc[:,0].values.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(handle_unknown='ignore')
y=ohe.fit_transform(y)
print(y)

folder='/home/clown/test_images/'

for file in listdir(folder):
    path=folder+file
    face_array=extract(path)
    face_array=np.expand_dims(face_array,axis=0)
    age=model.predict(face_array)
    out=ohe.inverse_transform(age)
    print(out)
    

    
    label=''
    if out==[[30]]:
        label='Middle_aged'
    elif out==[[15]]:
        label='Young'
    elif out==[[1]]:
        label='Child'        
    elif out==[[51]]:
        label='Old'

    output_array=detect(path,label)

    plt.imshow(output_array)
    plt.show()
    






