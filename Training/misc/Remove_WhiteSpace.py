import pickle
import re
import shutil
import os

path=os.getcwd()
pos=2
file = open(f'word_databse-{pos}.mf', 'rb') 
lo = pickle.load(file)
file.close()
lst={}
for i in lo.items():
    if(re.search(r'[\s]', i[1].strip()) is not None):
        lst[i[0]]=i[1]
        dest = shutil.move(f'{path}/words-{pos}/{i[0]}', f'{path}/words-{pos}_spaced/{i[0]}')

print(f'{len(lst)} words moved!!!')

for i in lst.items():
    del lo[i[0]]

file = open(f'word_databse-{pos}_notspaced.mf', 'wb') 
pickle.dump(lo, file)
file.close() 
file = open(f'word_databse-{pos}_spaced.mf', 'wb') 
pickle.dump(lst, file)                      
file.close() 