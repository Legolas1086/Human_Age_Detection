
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image


#Reading dataset
data=pd.read_csv('/home/clown/VS/Age_detection/age_gender.csv',delimiter=',')
data['pixels']=data['pixels'].apply(lambda x: np.array(x.split(),dtype='float32'))
x=np.array(data['pixels'].tolist())
x=x.reshape(x.shape[0],48,48)

#Converting images to RGB format
x_rgb=[]
for i in x:
    img=Image.fromarray(i)
    img=img.convert('RGB')
    img_array=np.asarray(img)
    x_rgb.append(img_array)


x_rgb=np.array(x_rgb)
print(x_rgb.shape)


#Encoding age 
y=data.iloc[:,0].values.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[0])])
y=ct.fit_transform(y)


#Splitting dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_rgb,y,test_size=0.2,random_state=0)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,GlobalAveragePooling2D,Dropout,BatchNormalization,Flatten
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam

#Using VGG16 as base
base_model=VGG16(weights='imagenet',include_top=False,input_shape=(48,48,3))
for layer in base_model.layers:
    layer.trainable=False


model=base_model.output
model=GlobalAveragePooling2D()(model)
model=Dense(64,activation='relu')(model)
model=Dropout(0.2)(model)
model=Dense(4,activation='softmax')(model)

final_model=Model(inputs=base_model.inputs,outputs=model)

#Compiling model
opt=Adam(lr=1e-3,decay=1e-3/25)
final_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
final_model.summary()


#Model Training
final_model.fit(x_train,y_train,validation_split=0.1,batch_size=8,epochs=25)

#Saving the Model 
final_model.save('age_detection.h5')


#Evaluating the model
eval=final_model.evaluate(x_test,y_test)
print(eval)

