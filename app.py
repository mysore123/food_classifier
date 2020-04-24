# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:34:57 2020

@author: apmys
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, request, redirect,url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

from tensorflow.keras.models import load_model
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.backend import set_session
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array,load_img


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    else:
        return render_template('index.html')


@app.route('/prediction/<filename>')
def prediction(filename):
    #Step 1
    print("Loading model")

    global model
    model = load_model('first_try.h5')

    #my_image = plt.imread(os.path.join('uploads', filename),format=(224,224))
    my_image=load_img(os.path.join('uploads',filename),target_size=(224,224))

    image = np.reshape(my_image,(224,224,3))
    print(image.shape)
    
    image = preprocess_input(image)
    print(image.shape)
    
   

    probabilities = model.predict(np.array( [image,] ))[0,:]
    print(probabilities)
#Step 4
    number_to_class = ['burger','cheesecake','chicken_wings','dosa','falafel','french_fries','fried_rice','garlic_bread','grilled_sandwich','hot_dog','ice_cream','idli','mysore pak','noodles','omelette','pizza','poha','puliyogre','samosa','vada','waffles']
    index = np.argsort(probabilities)
    print(index)
    
    calories={"dosa":'A dosa is a cooked flat thin layered rice batter, originating from South India, made from a fermented batter. Its main ingredients are rice and black gram that are grounded together in a fine, smooth batter with a dash of salt.  Average calories:133 Carbohydrates:75 calories protein:11 calories fat:47 calories',
              "idli":'Idlis are a type of savoury rice cake originating from south India.These are made by steaming a batter consisting of fermented black lentils.  Average Calories:33  carbs:29calories  proteins:4calories fat:1calorie',
              "mysore pak":'Mysore pak is an Indian sweet prepared in ghee that is popular in Southern India. It originated in the Indian state of Karnataka. It is made of generous amounts of ghee, sugar, gram flour, and often cardamom. Average Calories: 564 Carbohydrates: 68 calories, Proteins: 5 calories Fat: 491 calories',
              "poha":'Poha (flattened rice) is an easy, delicious and healthy breakfast recipe, popular in Maharashtra. Made with onions, potatoes and seasoning like chillies, lemon and curry leaves make up a tasty and easy meal of Poha. Average Calories: 180 (1 plate), Carbohydrates: 100calories, Proteins:9calories  Fat:71 calories.',
              }
    predictions = {
        "class1":number_to_class[index[20]],
        "class2":number_to_class[index[19]],
        "class3":number_to_class[index[18]],
        "prob1":probabilities[index[20]],
        "prob2":probabilities[index[19]],
        "prob3":probabilities[index[18]],
        "cal1":calories[number_to_class[index[20]]]
      }
#Step 5

    return render_template('predict.html', predictions=predictions)