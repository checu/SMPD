from tkinter import Tk, ttk, StringVar, Button, Label, Frame
from main import loadData
from main import *

root = Tk()
root.geometry("500x300")
root.config(background = "#F2DF95" )

# frame1=Frame(root, width=250, height=100, background="#0087BE")
# frame1.grid(row=0, column=0)
#
# frame2=Frame(root, width=250, height=100, background="#BE3700")
# frame2.grid(row=1, column=0)
#
# frame3=Frame(root, width=250, height=100, background="#BE9600")
# frame3.grid(row=3, column=0)
#
# frame4=Frame(root, width=250, height=150, background="#87BE00")
# frame4.grid(row=0, column=1)
# #
# frame5=Frame(root, width=250, height=150, background="#BE0087")
# frame5.grid(row=2, column=1)




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
    text.set(feature)
    # return feature

def calculate_SFS():
    combo_value = combo.get()
    FLD_averageMatrix()
    feature=SFS(int(combo_value))
    text.set(feature)


combo = ttk.Combobox(root)
combo.place(x=50,y=100)
combo['values'] = combo_values()
combo.current(0)


b = Button(root, text="Wczytaj z pliku",command=lambda:activbutton())
FSD_button=Button(root, text="FSD", state="disabled", command=lambda:calculate_FSD())
SFS_button=Button(root, text="SFS", state="disabled",command=lambda:calculate_SFS())

b.pack()
FSD_button.pack()
SFS_button.pack()


text=StringVar()
etykieta=Label(root,textvariable=text)
etykieta.pack()




root.mainloop()