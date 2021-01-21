## Let's Resize
import pickle
import cv2
import os
from PIL import Image

def resize_with_pad(im, target_width, target_height):
    target_ratio = target_height / target_width
    im_ratio = im.height / im.width
    if target_ratio > im_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * im_ratio)
    else:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / im_ratio)

    image_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background

path=os.getcwd()
for pos in range(1,4):
    file = open(f'word_databse-{pos}_notspaced.mf', 'rb') 
    lo = pickle.load(file)
    file.close()
    lst={}
    for i in lo.items():
        image = Image.open(f'wordsY-{pos}/{i[0]}', 'r')
        img = resize_with_pad(image,256,256)
        img.save(f'resizedY-{pos}/{i[0]}')
print('done!')

## Let's Crop

import cv2 as cv
import numpy as np

path=os.getcwd()
for pos in range(1,4):
    file = open(f'word_databse-{pos}_notspaced.mf', 'rb') 
    lo = pickle.load(file)
    file.close()
    lst={}
    for i in lo.items():
        img = cv.imread(f'words-{pos}/{i[0]}',cv.IMREAD_GRAYSCALE)
        if(img is None):
            print(i[0])
            continue
        gray = 255*(img < 128).astype(np.uint8) # To invert the text to white
        coords = cv.findNonZero(gray) # Find all non-zero points (text)
        x, y, w, h = cv.boundingRect(coords) # Find minimum spanning bounding box
        rect = img[y-5:y+h+5, x:x+w] # Crop the image - note we do this on the original image
        try:
            if(rect.shape[0]>30):
                cv.imwrite(f'wordsTrimmed-{pos}/{i[0]}', rect)
            else:
                cv.imwrite(f'wordsNotTrimmed-{pos}/{i[0]}', img)
                print(pos,'else->',i[0])
        except:
            cv.imwrite(f'wordsNotTrimmed-{pos}/{i[0]}', img)
            print(pos,'except->',i[0])
