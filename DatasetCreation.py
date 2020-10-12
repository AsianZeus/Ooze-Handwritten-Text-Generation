import os
# from zipfile import ZipFile 

# path = os.getcwd()+'/Desktop/Dataset'
# file_name = path+"/ascii/words.txt"

# wordfile= open(file_name,'r')
# f1= wordfile.readlines()
# lines=[]
# for x in f1:
#     lines.append(x)
# print(lines[0])
# image_filename={}
# for i in lines:
#     word=i.split(' ')
#     image_filename[word[0]] = word[-1][:-1]
# print(image_filename)

import pickle
# file = open('metafile.mf', 'ab') 
# pickle.dump(image_filename, file)                      
# file.close() 

file = open('metafile.mf', 'rb') 
meta = pickle.load(file) 
file.close() 

file = open('metafileupdated.mf', 'rb') 
metaupdated = pickle.load(file) 
file.close() 

# print(meta)


# path = os.getcwd()+'/Desktop/Dataset/Data'
# filename = os.listdir(path)
# ctr=1
# for name in filename:
#     try:
#         os.rename(path+f'/{name[:-4]}.png', path+f'/{meta[name[:-4]]}_{name[:-4]}.png')
#         ctr+=1
#     except:
#         pass
# print(ctr,'files updated!')

# updatedfiles=[]
# path = os.getcwd()+'/Desktop/Dataset/Data'
# filename = os.listdir(path)
# for name in filename:
#     updatedfiles.append(name[:-4])

import shutil
path = os.getcwd()+'/Desktop/Dataset/Data'
for i in metaupdated:
    if(i.find('_')==-1):
        shutil.move(f"{path}/{i}.png", f"{path}/notupdated")