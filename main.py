from math import sqrt

import numpy
from numpy import *
from sympy import *
import itertools

# Obiekt do trzymania informacji o probce
class Sample:
    def __init__(self, name, identifier, features):
        self.name = name
        self.identifier = identifier
        self.features = features

    # Zwraca ilosc posiadanych cech
    def featuresCount(self):
        return len(self.features)

    # Zwraca tablice cech
    def getFeatures(self):
        return self.features

    # Zwraca nazwe probki Acer lub Quercus
    def getName(self):
        return self.name


# Ilosc probek o nazwie Acer
Ac = 0
# Ilosc probek o nazwie Quercus
Qc = 0

# Otwiera plik txt i parsuje wszystkie linijki do obiektow
def loadData(filename):
    # Tablica wszystkich probek
    samples = []
    with open(filename) as f:
        for line in f:
            data = line.split(",")
            # Pierwszy czlon nazwy probki
            name = data[0].split(" ")[0]
            # Identyfikator 0 = Quercus , 1 = Acer
            identifier = 0
            # tablica 64 elementowa
            features = [None] * 64
            if "Acer" in name:
                identifier = 1
                global Ac
                Ac += 1
            else:
                global Qc
                Qc += 1
            # Petla po wszystkich cechach
            for feature in range(1, 65):#65
                # Dodajemy do tablicy cech wartosc cechy w postaci float
                features[feature - 1] = float(data[feature])
            # Tworzymy obiekt probki i dodajemy go do tablicy wszystkich probek
            samples.append(Sample(name, identifier, features))
    return samples
#---------------------cześć dodana------------------------------------------------------
#---------------------podziała danych na swie klasy-------------------------------------
ACER=[]
QUERTUS=[]

def get_classes(samples):
    global ACER
    global QUERTUS
    for object in samples:
        if object.getName() == "Acer":
            ACER.append(object.getFeatures())
        else:
            QUERTUS.append((object.getFeatures()))



get_classes(loadData("data.txt"))
#--------------------------------
# Fisher Single Dimension
def FSD(samples):
    FLD = 0
    tmp = 0
    index = -1
    #acrr = getTupleOfCount(samples)
    Ac = 0
    Qc = 0

    for i in range(0,64):
        averageAcer = 0
        averagesQuercus = 0
        standardAcer = 0
        standardQuercus = 0

        for object in samples:
            if object.getName() == "Acer":
                Ac += 1
                averageAcer += object.getFeatures()[i]
                standardAcer += object.getFeatures()[i] * object.getFeatures()[i]
            else:
                Qc +=1
                averagesQuercus += object.getFeatures()[i]
                standardQuercus += object.getFeatures()[i] * object.getFeatures()[i]

        averageAcer /= Ac
        averagesQuercus /= Qc
        standardAcer = standardAcer / Ac - averageAcer * averageAcer
        standardQuercus = standardQuercus / Qc - averagesQuercus * averagesQuercus

        tmp = abs(averageAcer - averagesQuercus) / (sqrt(standardAcer) + sqrt(standardQuercus))
        if tmp > FLD:
            FLD = tmp
            index = i
        Ac = 0
        Qc = 0

    return index
#------------------------------------------multi dimentions Fisher---------------------------------------------------------------------
Aceraverage = []
Quercusaverage = []
# liczenie średniej wartości dla każdej cechy
def FLD_averageMatrix():
    global Aceraverage
    global Quercusaverage
    for rowA in ACER:
        Aceraverage.append(sum(rowA)/len(rowA))
    for rowQ in QUERTUS:
        Quercusaverage.append(sum(rowQ) / len(rowQ))

#FLD_averageMatrix(loadData("proba.txt"))#"data.txt"
#-----------------------------------Fisher wielowymiarowy-----------------------------------
def FLD_listOfcombination(n):
    FLD=float(0)
    index_list=[]
    combinations= itertools.combinations(range(64), n)
    for combination in combinations:
        print(combination)

        temp=Fisher(combination)

        print(temp)

        if temp>FLD:
            FLD=temp
            index_list=combination
    listOfIndex = [x + 1 for x in list(index_list)]# zwiekszone o 1
    print("max:",listOfIndex)
    print (FLD)
    return list(index_list)


def Fisher(combination):
    matrixAC = []
    matrixQR = []
    AC_avr_vector = []
    QR_avr_vector = []
    # print(Quercusaverage)
    # #print("wektor,", QUERTUS[element]-Quercusaverage[element])
    for element in combination:
        value_average_AC = numpy.subtract(ACER[element],Aceraverage[element])
        value_average_QR = numpy.subtract(QUERTUS[element],Quercusaverage[element])
        matrixAC.append(value_average_AC)
        matrixQR.append(value_average_QR)
        AC_avr_vector.append(Aceraverage[element])
        QR_avr_vector.append(Quercusaverage[element])

    covariation_matrix_AC=numpy.dot(numpy.array(matrixAC), numpy.array(matrixAC).transpose())
    covariation_matrix_QR=numpy.dot(numpy.array(matrixQR), numpy.array(matrixQR).transpose())

    det_AC=numpy.linalg.det(covariation_matrix_AC)
     #print("deatAc:",det_AC)
    det_QR=numpy.linalg.det(covariation_matrix_QR)
     #print("deatQR:", det_QR)

    absolut=numpy.subtract(AC_avr_vector,QR_avr_vector)

    Fisher=numpy.divide(numpy.linalg.norm(absolut),(det_AC+det_QR))

    return Fisher


#----------------------------------------SFS------------------------------------
def SFS(steps):
    SFS=0
    best_features_index=[]
    FLD_averageMatrix()#wrzucic do guzika
    best_state_table=[]
    samples=loadData("data.txt")
    feature_list = list(range(64))
    for step in range (1,steps+1):
        print(best_state_table)
        if step==1:
            first = FSD(samples)
            best_state_table.append(first)
            best_features_index=best_state_table
        else:
            feature_list=[x for x in feature_list if x not in best_state_table]
            for element in feature_list:
                best_state_table.append(element)
                # print(best_state_table)
                temp_fisher=Fisher(best_state_table)
                if temp_fisher>SFS:
                    best_features_index = list(best_state_table)
                    SFS=temp_fisher
                # print (temp_fisher)
                best_state_table.pop(-1)
                # print (best_state_table)
            best_state_table.append(best_features_index[-1])
    print(best_features_index)
    return best_features_index
SFS(3)




# FLD_listOfcombination(2)

# return Tuple of Acer Samples Count and Quercus samples Count
def getTupleOfCount(samples):
    Acount = 0
    Qcount = 0
    for object in samples:
        if object.getName() == "Acer":
            Acount += 1

        else:
            Qcount += 1

    return (Acount,Qcount)

# FLD_listOfcombination(62)
#print(loadData("data.txt"))
#FSD()

