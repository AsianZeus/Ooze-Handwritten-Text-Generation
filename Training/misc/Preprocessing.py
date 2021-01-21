import numpy as np
import cv2
import os
import pickle
path = 'C:/Users/Ace/Desktop/Sem 3 Project/HTG/Dataset/Handwritten-Text-Generation'
os.chdir(path)

#Find Small width lines

file = open('metafile.mf', 'rb') 
meta = pickle.load(file)
file.close() 
RemovedFiles=[]
for i in meta.items():
    img = cv2.imread(path+'/DataAr/'+i[0]+'.png')
    h,w,c=img.shape
    if(w<200):
        RemovedFiles.append(i[0])

file = open('metaremoved.mf', 'ab') 
pickle.dump(RemovedFiles, file)                      
file.close() 
print(len(RemovedFiles),'\n',RemovedFiles)


#Delete File

file = open('metaremoved.mf', 'rb') 
meta = pickle.load(file) 
file.close() 

print(len(meta))
for i in meta:
    os.remove(path+'/DataAr/'+i+'.png')
    os.remove(path+'/Data/'+i+'.png')
#    del meta[i]
    print(i,'File Deleted!')    


#Resize and Grayscale

file = open('metafile.mf', 'rb') 
meta = pickle.load(file) 
file.close()
for i in meta.items():
    img=cv2.imread(path+'/Data/'+i[0]+'.png',cv2.IMREAD_GRAYSCALE)
    imgaug=cv2.imread(path+'/DataAr/'+i[0]+'.png',cv2.IMREAD_GRAYSCALE)
    # print(img.shape,imgaug.shape)
    img = cv2.resize(img, (512, 48)) 
    imgaug = cv2.resize(imgaug, (512, 48))
    cv2.imwrite(path+'/PData/'+i[0]+'.png',img)
    cv2.imwrite(path+'/PDataAr/'+i[0]+'.png',imgaug) 
    # cv2.imshow("img",img)
    # cv2.imshow("imgaug",imgaug)
    # cv2.waitKey(0)
    # break
print('Done!!')