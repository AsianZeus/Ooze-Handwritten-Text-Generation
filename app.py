from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from flask import Flask, redirect, url_for, render_template, request, make_response
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

app = Flask(__name__)
g_model = load_model('model.h5',custom_objects={'InstanceNormalization':InstanceNormalization})

count=0
@app.route('/')
def hello_world():
    files = os.listdir('static')
    for i in files:
        if(i != 'style.css'):
            os.remove(f'static/{i}')
    print(files)
    return render_template("index.html") 

def text2str(text):
    img = Image.new('RGB', (900, 800),color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 50)
    d.text((0,0), text, font=font,fill=(0,0,0))
    text_width, text_height = d.textsize(text,font=font)
    open_cv_image = np.array(img)
    image = open_cv_image[:, :, ::-1].copy()[0:text_height+3,0:text_width]
    return image

def ValuePredictor(text):
    global count
    global g_model
    farray=[]
    length=len(text)
    for i in range(0,length,3):
        if(i+3>length):
            val=text[i:length]
        else:
            val=text[i:i+3]
        temp = text2str(val)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

        temp = cv2.resize(temp, (256, 256))

        temp = (temp - 127.5) / 127.5
        img = temp.reshape((1,256,256,1))
        Ximg = g_model.predict(img)
        img = (Ximg+1) / 2.0

        img = img.reshape((256,256))
        farray.append(img)
    imx=cv2.hconcat(farray)
    print('0-->>>',imx.shape)
    fname=f'image_{count}.png'
    cv2.imwrite(f'static/{fname}',cv2.convertScaleAbs(imx, alpha=(255.0)))
    count+=1
    return fname

@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        dict_list = request.form.to_dict()
        ftext = dict_list['str']
        prediction=ValuePredictor(ftext)
        return render_template("result.html", prediction = prediction)

if __name__ == '__main__':
    
    app.run(debug=True)