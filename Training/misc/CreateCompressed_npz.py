from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

def load_images(path, size=(256,256)):
	data_list = list()
	for filename in listdir(path):
		pixels = load_img(path + filename, target_size=size,color_mode="grayscale")
		pixels = img_to_array(pixels)
		data_list.append(pixels)
	return asarray(data_list)

resized1 = load_images('resized-1/')
resized2= load_images('resized-2/')
resized3 = load_images('resized-3/')
X = vstack((resized1, resized2, resized3))
print('Loaded dataA: ', resized1.shape,resized2.shape,resized3.shape)

resizedY1 = load_images('resizedY-1/')
resizedY2= load_images('resizedY-2/')
resizedY3 = load_images('resizedY-3/')
Y = vstack((resizedY1, resizedY2, resizedY3))
print('Loaded dataA: ', resizedY1.shape,resizedY2.shape,resizedY3.shape)

filename = 'words2handwritingresize.npz'
savez_compressed(filename, X,Y)
print('Saved dataset: ', filename)

#Verification
# import numpy as np
# data = np.load('words2handwritingresize.npz')
# trainX, trainY = data['arr_0'], data['arr_1']
# print(trainX.shape, trainY.shape)