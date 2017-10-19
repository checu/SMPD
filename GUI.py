from tkinter import Tk, ttk, StringVar, Button, Label, Frame
from main import loadData
from main import *

root = Tk()
root.geometry("500x300")
root.config(background = "#F2DF95" )

frame1=Frame(root, width=250, height=100, background="#0087BE", relief='groove',bd=2)
frame1.place(x=0,y=0)

frame2=Frame(root, width=250, height=100, background="#BE3700", relief='groove',bd=2)
frame2.place(x=0,y=100)
#
frame3=Frame(root, width=250, height=100, background="#BE9600", relief='groove',bd=2)
frame3.place(x=0,y=200)
#
frame4=Frame(root, width=250, height=150, background="#87BE00", relief='raised',bd=2)
frame4.place(x=250,y=0)
# #
frame5=Frame(root, width=250, height=150, background="#006F81", relief='raised',bd=2)
frame5.place(x=250,y=150)




# frame = Frame(root,width=150, height= 75,bg="#ddc258")
# frame.place(relx=0,rely=0)

#frame.pack()

tablicaProbek = []

def combo_values():
    tablica=[]
    for i in range(1,65):
        tablica.append(i)
    return tablica

# countryvar = StringVar()
def activbutton():
    global tablicaProbek
    tablicaProbek = loadData("data.txt")
    FSD_button['state']= "normal"
    SFS_button['state'] = "normal"
    return tablicaProbek

def calculate_FSD():
    combo_value=combo.get()
    if int(combo_value)==1:
        feature=FSD(tablicaProbek)
    else:
        FLD_averageMatrix()
        feature=FLD_listOfcombination(int(combo_value))
    text_FSD.set(feature)
    # return feature

def calculate_SFS():
    combo_value = combo.get()
    FLD_averageMatrix()
    feature=SFS(int(combo_value))
    text_SFS.set(feature)


combo = ttk.Combobox(root)
combo.place(x=35,y=60)
combo['values'] = combo_values()
combo.current(0)


b = Button(root, text="Wczytaj z pliku",background="#006C98",fg='#EAE4CC',command=lambda:activbutton())
FSD_button=Button(root, text="FSD", state="disabled",background="#A63000",fg='#EAE4CC',command=lambda:calculate_FSD())
SFS_button=Button(root, text="SFS", state="disabled", background="#987800",fg='#EAE4CC',command=lambda:calculate_SFS())

b.place(x=35,y=25)
FSD_button.place(x=35,y=130)
SFS_button.place(x=35,y=230)


text_FSD=StringVar()
text_SFS=StringVar()

titie_general=Label(root,text="GENERAL TAB" ,relief='groove')
title_FSD=Label(root,text="FISHER" ,relief='groove')
titile_SFS=Label(root,text="Sequential Forward Selection" ,relief='groove')

etykietaFSD=Label(root, textvariable=text_FSD,relief='groove')
etykietaSFS=Label(root,text="SFS:",textvariable=text_SFS,relief='groove')


titie_general.place(x=0,y=0)
title_FSD.place(x=0,y=100)
titile_SFS.place(x=0,y=200)
etykietaFSD.place(x=100,y=132.5)
etykietaSFS.place(x=100,y=232.5)




root.mainloop()