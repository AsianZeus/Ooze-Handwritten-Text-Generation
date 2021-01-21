import pickle
from tkinter import *
import tkinter.ttk as tpx
from PIL import Image, ImageTk, ImageOps
import os
import cv2

def key(event):
    nextImage()


def wordlist(i):
    s='_'
    x=i.split('_')
    name=s.join(x[:4])+'.png'
    # print(name)
    words=meta1[name].split(' ')
    return words


def nextImage():
    if(imgcounter>=totalfiles):
        quit()

    imgpath=globals()['path']+str(FileName[globals()['imgcounter']])
    imgnew= cv2.imread(imgpath)
    
    metalabel[FileName[globals()['imgcounter']]]= str(entryText.get())

    globals()['imgcounter']+=1
    globals()['cnt']+=1

    Lb1.delete(0,'end')
    try:
        print(FileName[globals()['imgcounter']])
        xox=wordlist(FileName[globals()['imgcounter']])
        for i in range(len(xox)):
            Lb1.insert(i+1, xox[i])
    except:
        filx= open('word_database1.mf','wb')
        pickle.dump(metalabel,filx)
        filx.close()
        print('closed!!!')
        quit()

    textlabel.delete(0,END)
    
    try:
        entryText.set(loadedwords[FileName[globals()['imgcounter']]])
    except:
        pass
    
    try:
        imo=Image.open(globals()['path']+FileName[globals()['imgcounter']])
        width=imo.size[0]
        height=imo.size[1]
        imgtk = ImageTk.PhotoImage(ImageOps.fit(imo, (width,height)))
        imglabel.configure(image=imgtk,height=height,width=width)
        imglabel.image = imgtk
        imglabel.grid(column=0, row=0,padx=0)
    except:
        pass
    
def previousImage():
    imgpath=globals()['path']+str(FileName[globals()['imgcounter']])
    imgnew= cv2.imread(imgpath)
    
    metalabel[FileName[globals()['imgcounter']]]= str(entryText.get())
    

    globals()['imgcounter']-=1
    globals()['cnt']-=1

    Lb1.delete(0,'end')
    try:
        print(FileName[globals()['imgcounter']])
        xox=wordlist(FileName[globals()['imgcounter']])
        for i in range(len(xox)):
            Lb1.insert(i+1, xox[i])
    except:
        filx= open('word_database1.mf','wb')
        pickle.dump(metalabel,filx)
        filx.close()
        print('closed!!!')
        quit()

    textlabel.delete(0,END)
    
    try:
        entryText.set(loadedwords[FileName[globals()['imgcounter']]])
    except:
        pass
    
    try:
        imo=Image.open(globals()['path']+FileName[globals()['imgcounter']])
        width=imo.size[0]
        height=imo.size[1]
        imgtk = ImageTk.PhotoImage(ImageOps.fit(imo, (width,height)))
        imglabel.configure(image=imgtk,height=height,width=width)
        imglabel.image = imgtk
        imglabel.grid(column=0, row=0,padx=0)
    except:
        pass


def suggestions(evt):
    passvalue=str((Lb1.get(Lb1.curselection())))
    entryText.set(passvalue)


path=os.getcwd()+'/words-1/'

FileName= os.listdir(path)
totalfiles=len(FileName)

metalabel={}
filer = open('meta1.mf', 'rb')
meta1=pickle.load(filer)
filer.close()

imgcounter=0
cnt=0
loadedwords={}


r = Tk() 
r.configure(background='black')
r.geometry("800x350")
r.title('Dataset Creator')

label = Label(r,text="Enter Label:",bg='black',fg='white')
label.grid(column=0,row=1,padx=5, pady=5,sticky=W)

entryText = StringVar()
textlabel = Entry(r,textvariable=entryText)
textlabel.grid(column=0,row=1,padx=100, pady=5,sticky=W)
textlabel.bind('<Return>',key)



try:
    filb= open('word_databse-1.mf','rb')
    loadedwords=pickle.load(filb)
    print('Words Loaded')
except:
    print('starting from the start')
try:
    entryText.set(loadedwords[FileName[imgcounter]])
except:
    pass

nextbtn = Button(r,text="Next",command=nextImage)
nextbtn.place(x=150, y=200)
# nextbtn.grid(column=0, row=2,padx=5, pady=5,sticky=E)

backbtn = Button(r,text="Back",command=previousImage)
# backbtn.grid(column=0, row=2,padx=5, pady=5,sticky=W)
backbtn.place(x=20, y=200)

Lb1 = Listbox(r)
xox=wordlist(FileName[imgcounter])
for i in range(len(xox)):
    Lb1.insert(i+1, xox[i])
Lb1.grid(column=1, row=0,padx=50,rowspan=3,sticky=W)
Lb1.bind('<<ListboxSelect>>',suggestions)
imo=Image.open(path+FileName[imgcounter])
width=imo.size[0]
height=imo.size[1]
imglabel = Label(r,height=height,width=width)
imglabel.configure(background='black')

imgtk = ImageTk.PhotoImage(ImageOps.fit(imo, (width,height)))
imglabel.configure(image=imgtk)
imglabel.image = imgtk
imglabel.grid(column=0, row=0,padx=0)
r.mainloop()
try:
    filb.close()
except:
    pass
print('closed!')
filx= open('word_database1.mf','wb')
pickle.dump(metalabel,filx)
filx.close()