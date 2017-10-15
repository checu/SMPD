from tkinter import Tk, ttk, StringVar, Button, Label
from main import loadData
from main import *

root = Tk()

tablicaProbek = []

def combo_values():
    tablica=[]
    for i in range(1,65):
        tablica.append(i)
    return tablica

# countryvar = StringVar()

def calculate_FSD():
    combo_value=combo.get()
    if int(combo_value)==1:
        feature=FSD(tablicaProbek)
    else:
        feature=FLD_multi_feature(tablicaProbek,int(combo_value))
    return feature


def activbutton():
    global tablicaProbek
    tablicaProbek = loadData("data.txt")
    b_licz['state']="normal"


combo = ttk.Combobox(root)
combo.place(x=50,y=100)
combo['values'] = combo_values()
combo.current(0)


b = Button(root, text="Wczytaj z pliku",command=lambda:activbutton())
b_licz=Button(root, text="Licz", state="disabled", command=lambda:text.set(calculate_FSD()))#text.set(FSD(tablicaProbek))
b.pack()
b_licz.pack()


text=StringVar()
etykieta=Label(root,textvariable=text)
etykieta.pack()



root.geometry("300x300")
root.mainloop()