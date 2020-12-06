from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from flask import Flask, redirect, url_for, render_template, request, make_response
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import cv2
import os
import gc

g_model = tf.keras.models.load_model('model.h5',custom_objects={'InstanceNormalization':InstanceNormalization},compile=False)

app = Flask(__name__)

count=0
@app.route('/')
def hello_world():
    files = os.listdir('static')
    for i in files:
        if(i != 'style.css'):
            os.remove(f'static/{i}')
    return render_template("index.html") 

def text2str(text):
    img = Image.new('RGB', (900, 800),color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("font/arial.ttf", 50)
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
        del Ximg, img, temp
        gc.collect()
    imx=cv2.hconcat(farray)
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
    app.run(debug=False)