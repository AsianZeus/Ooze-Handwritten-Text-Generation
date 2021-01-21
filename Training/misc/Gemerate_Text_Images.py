from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import pickle

def text2str(text):
    img = Image.new('RGB', (900, 800),color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 50)
    d.text((0,0), text, font=font,fill=(0,0,0))
    text_width, text_height = d.textsize(text,font=font)
    open_cv_image = np.array(img)
    image = open_cv_image[:, :, ::-1].copy()[0:text_height+3,0:text_width]
    return image


path=os.getcwd()
for pos in range(3,4):
    file = open(f'word_databse-{pos}_notspaced.mf', 'rb') 
    lo = pickle.load(file)
    file.close()
    for i in lo.items():
        try:
            image=text2str(f"{i[1].strip()}")           #{ 'a01-002-152.png': 'A horse like strcture!' }
            cv2.imwrite(f"{os.getcwd()}/wordsY-{pos}/{i[0]}",image)
        except:
            print(i[0])
print('done!')

