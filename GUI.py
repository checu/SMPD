from tkinter import Tk, ttk, StringVar, Button, Label, Frame, Entry
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


def train():
    trainig_part = ent_trainig_part.get()#wartości od 0 do 1
    get_Test_Training_Matrix(float(trainig_part))
    # return trainig_part

def calculate_clasyficators(): #execute
    clasyficator=combo_clas.get()
    k=int(ent_k_part.get())
    efficiency=clasyficator_calculation(clasyficator,k)
    text_cal_ef.set(efficiency)

#combosy
combo = ttk.Combobox(root)
combo.place(x=35,y=60)
combo['values'] = combo_values()
combo.current(0)

combo_clas = ttk.Combobox(root)
combo_clas.place(x=350,y=72)
combo_clas['values'] = ["NN","k-NN","NM","k-NM"]

#entry pola
ent_trainig_part=Entry(root)
ent_trainig_part.place(x=350,y=42)
ent_bootstrap_iter=Entry(root)
ent_bootstrap_iter.place(x=350,y=192)
ent_crosvalid_range=Entry(root)
ent_crosvalid_range.place(x=350,y=232)
k=StringVar()
ent_k_part=Entry(root,width=5,textvariable=k)
k.set(0)
ent_k_part.place(x=300,y=100)

#guziki
b = Button(root, text="Wczytaj z pliku",background="#006C98",fg='#EAE4CC',command=lambda:activbutton())
FSD_button=Button(root, text="FSD", state="disabled",background="#A63000",fg='#EAE4CC',command=lambda:calculate_FSD())
SFS_button=Button(root, text="SFS", state="disabled", background="#987800",fg='#EAE4CC',command=lambda:calculate_SFS())
train_button= Button(root, text="Train",background="#5E8500",fg='#EAE4CC', command=lambda:train())
execute_button=Button(root, text="Execute",background="#5E8500",fg='#EAE4CC',command=lambda:calculate_clasyficators())
bootstrap=Button(root, text="Bootstrap",background="#006C98",fg='#EAE4CC')
crosvalidation=Button(root, text="Crosval",background="#006C98",fg='#EAE4CC')


b.place(x=35,y=25)
FSD_button.place(x=35,y=130)
SFS_button.place(x=35,y=230)
train_button.place(x=285,y=40)
execute_button.place(x=285,y=70)
bootstrap.place(x=285,y=190)
crosvalidation.place(x=285,y=230)


#etykiety tytuły
titie_general=Label(root,text="GENERAL TAB" ,relief='groove')
title_FSD=Label(root,text="FISHER" ,relief='groove')
titile_SFS=Label(root,text="Sequential Forward Selection" ,relief='groove')
titile_clasificators=Label(root,text="Klasyfikatory" ,relief='groove')
title_qualityclasification=Label(root,text="Jakość klasyfikacji" ,relief='groove')

#etykiety wyswietlajace wyniki

text_FSD=StringVar()
text_SFS=StringVar()
text_cal_ef=StringVar()
text_quality_ef=StringVar()

etykietaFSD=Label(root, textvariable=text_FSD,relief='groove')
etykietaSFS=Label(root,textvariable=text_SFS,relief='groove')
etykierta_clas_eff=Label(root,textvariable=text_cal_ef,relief='groove')
etykierta_quality_eff=Label(root,textvariable=text_quality_ef,relief='groove')

subtitiel_cal=Label(root,text="%",relief='flat',fg='#EAE4CC',bg="#87BE00")
training_part=Label(root,text="Training part",relief='flat',fg='#EAE4CC',bg="#87BE00")
k_clasyficators=Label(root,text="k:",relief='flat',fg='#EAE4CC',bg="#87BE00")
subtitiel_cal_quality=Label(root,text="%",relief='flat',fg='#EAE4CC',bg="#006F81")



#geometria
titie_general.place(x=0,y=0)
title_FSD.place(x=0,y=100)
titile_SFS.place(x=0,y=200)
titile_clasificators.place(x=250,y=0)
title_qualityclasification.place(x=250,y=150)
etykietaFSD.place(x=100,y=132.5)
etykietaSFS.place(x=100,y=232.5)
etykierta_clas_eff.place(x=350,y=100)
etykierta_quality_eff.place(x=350,y=260)
subtitiel_cal.place(x=400,y=100)
training_part.place(x=350,y=20)
subtitiel_cal_quality.place(x=400,y=260)
k_clasyficators.place(x=285,y=100)


root.mainloop()