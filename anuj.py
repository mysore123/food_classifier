# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:31:43 2020

@author: apmys
"""


import numpy as np
import os
import time
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#%%
#%%

# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
#		x = x/255
#		print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 21
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:1000]=0
labels[1000:2000]=1
labels[2000:3000]=2
labels[3000:3988]=3
labels[3988:4986]=4
labels[4986:5986]=5
labels[5986:6986]=6
labels[6986:7986]=7
labels[7986:8986]=8
labels[8986:9985]=9
labels[9985:10984]=10
labels[10984:11964]=11
labels[11964:12805]=12
labels[12805:13865]=13
labels[13865:14865]=14
labels[14865:15865]=15
labels[15865:16386]=16
labels[16386:16951]=17
labels[16951:17951]=18
labels[17951:18973]=19
labels[18973:19973]=20


names = ['burger','cheesecake','chicken_wings','dosa','falafel','french_fries','fried_rice','garlic_bread','grilled_sandwich','hot_dog','ice_cream','idli','mysore pak','noodles','omelette','pizza','poha','puliyogre','samosa','vada','waffles']
 
# convert class labels to on-hot encoding
Y = to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#########################################################################################
# Custom_vgg_model_1
#Training the classifier alone
image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()
last_layer = model.get_layer('fc2').output
#x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

for layer in custom_vgg_model.layers[:-1]:
	layer.trainable = False

custom_vgg_model.layers[3].trainable

custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


t=time.time()
#%%Train
t=time.time()
hist = custom_vgg_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
custom_vgg_model.save('first_try.h5')
#%% feature extraction also


#%%Plotting loss and accuracy

import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(12)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])

#%% Prediction
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from numpy import loadtxt
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array

newmodel=load_model('first_try.h5')

image = load_img('test/mysorepak.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
print(image.shape)

# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
print(image.shape)
    
image = preprocess_input(image)
print(image.shape)

yhat = newmodel.predict(image)
print(yhat)

if yhat[0][0]>=0.5:
    print('Burger')
    print(yhat[0][0]*100,'%')
    np.set_printoptions(suppress=True)
    
elif  yhat[0][1]>=0.5:
    print('Cheesecake')
    print(yhat[0][1]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][2]>=0.5:
    print('Chicken Wings')
    print(yhat[0][2]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][3]>=0.5:
    print('Dosa')
    print(yhat[0][3]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][4]>=0.5:
    print('Falafel')
    print(yhat[0][4]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][5]>=0.5:
    print('French Fries')
    print(yhat[0][5]*100,'%')
    np.set_printoptions(suppress=True)

elif yhat[0][6]>=0.5:
    print('Fried Rice')
    print(yhat[0][6]*100,'%')
    np.set_printoptions(suppress=True)

elif yhat[0][7]>=0.5:
    print('Garlic Bread')
    print(yhat[0][7]*100,'%')
    np.set_printoptions(suppress=True)

elif yhat[0][8]>=0.5:
    print('Sandwich')
    print(yhat[0][8]*100,'%')
    np.set_printoptions(suppress=True)

elif yhat[0][9]>=0.5:
    print('Hot dog')
    print(yhat[0][9]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][10]>=0.5:
    print('Ice cream')
    print(yhat[0][10]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][11]>=0.5:
    print('Idli')
    print(yhat[0][11]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][12]>=0.5:
    print('Mysore Pak')
    print(yhat[0][12]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][13]>=0.5:
    print('Noodles')
    print(yhat[0][13]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][14]>=0.5:
    print('Omelette')
    print(yhat[0][14]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][15]>=0.5:
    print('Pizza')
    print(yhat[0][15]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][16]>=0.5:
    print('Poha')
    print(yhat[0][16]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][17]>=0.5:
    print('Puliyogre')
    print(yhat[0][17]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][18]>=0.5:
    print('Samosa')
    print(yhat[0][18]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][19]>=0.5:
    print('Vada')
    print(yhat[0][19]*100,'%')
    np.set_printoptions(suppress=True)
        
else:
    print('Waffles')
    print(yhat[0][20]*100,'%')
    np.set_printoptions(suppress=True)