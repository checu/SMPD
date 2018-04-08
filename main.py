from math import sqrt

import numpy

from numpy import *
from sympy import *
import itertools
from random import *
import time
import heapq


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


    ACER=numpy.array(ACER).transpose()
    print(ACER)
    QUERTUS=numpy.array(QUERTUS).transpose()

get_classes(loadData("data.txt"))
#-----------------------------------------------------dane do klasyfikacji-----------------------------
SFS_index=[]
Train_Test_dictionary={}

def get_Test_Training_Matrix(part):
    global Train_Test_dictionary
    ACER_clas= ACER.tolist()
    QUERTUS_clas=QUERTUS.tolist()
    ACER_manipultaion_matrix=[]
    QUERTUS_manipulation_matrix=[]

    for n in SFS_index:
        ACER_manipultaion_matrix.append(ACER_clas[n])
        QUERTUS_manipulation_matrix.append(QUERTUS_clas[n])


    friction_acer=int(part*len(ACER_clas[0]))
    friction_quertus=int(part*len(QUERTUS_clas[0]))
    
    #generowanie losowych indexow do odziału- test- trening
    index_lista_acer=range(0,len(ACER_clas[0])-1)
    index_lista_quertus=range(0,len(QUERTUS_clas[0])-1)
    training_Acer=sample(index_lista_acer,friction_acer)
    training_Quetus= sample(index_lista_quertus,friction_quertus)


    #tworzenie tablica treningowa / testowa:
    Acer_training_matrix=[[] for i in range(len(ACER_manipultaion_matrix))]
    Quertus_training_matrix=[[] for i in range(len(QUERTUS_manipulation_matrix))]
    Acer_test_matrix=[]
    Quertus_test_matrix=[]

    # przypisanie odpowiednich wartosci do macierzy
    for x in range(0,len(ACER_manipultaion_matrix)):
        for index in training_Acer:
            z=ACER_manipultaion_matrix[x][index]
            Acer_training_matrix[x].append(z)
        Acer_test_matrix.append([m for i, m in enumerate(ACER_manipultaion_matrix[x]) if i not in training_Acer])

    # zyta=Acer_test_matrix

    # print(Acer_test_matrix)
    for y in range(0,len(QUERTUS_manipulation_matrix)):
        for index in training_Quetus:
            q=QUERTUS_manipulation_matrix[y][index]
            Quertus_training_matrix[y].append(q)
        Quertus_test_matrix.append([d for i, d in enumerate(QUERTUS_manipulation_matrix[y]) if i not in training_Quetus])

    #tworzenie jednej tablicy probek testowych
    Combine_test_matrix = [[] for f in range(len(Acer_test_matrix))]

    #print("dlugosc",len(Acer_test_matrix))
    #print("Acer:",len(Acer_test_matrix[0]))

    for t in range(0,len(Acer_test_matrix)):
        Combine_test_matrix[t]=Acer_test_matrix[t]+Quertus_test_matrix[t]

    # print("acerTest,len",len(Acer_test_matrix[0]))
    # print("combine matrix",len(Combine_test_matrix[0]))
    # koniec tworzenia wspolej tablicy probek
    Train_Test_dictionary= {"ACER_Training":Acer_training_matrix,"Quertus_Trainig":Quertus_training_matrix,"ACER_Test":Acer_test_matrix,"Quertus_Test":Quertus_test_matrix,"Combine_Test":Combine_test_matrix}

    # return {"ACER_Training":Acer_training_matrix,"Quertus_Trainig":Quertus_training_matrix,"ACER_Test":Acer_test_matrix,"Quertus_Test":Quertus_test_matrix,"Compbine_Test":Combine_test_matrix}
# czy mozna zrobic return as global?

get_Test_Training_Matrix(0.1)
#SFS(3)
#-----------------------------------------
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
    i=0
    combinations= itertools.combinations(range(64), n)
    for combination in combinations:
        i=i+1
        print(combination)

        temp=Fisher(combination)

        print(temp)

        if temp>FLD:
            FLD=temp
            index_list=combination
    listOfIndex = [x + 1 for x in list(index_list)]# zwiekszone o 1
    print("i",i)
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
    global SFS_index
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
    SFS_index=best_features_index
    return best_features_index

SFS(3)
get_Test_Training_Matrix(0.2)



#--------------------------------------NN----------------------------------------------------
    # Klasyfikatory
    # if clasyficator=="NN":
    #     NN_good_samples = len(Combine_Test[0])
        # for test_vect in range(0,len(Combine_Test[0])):
        #     A_min_dist=1000
        #     Q_min_dist = 1000
        #     for A_train_vect in range(0,len(ACER_Training[0])):
        #         A_euqlidean_distance=0
        #         A_suma_fin=0
        #         for element in range(0,len(Combine_Test)):
        #             A_suma=((ACER_Training[element][A_train_vect])-Combine_Test[element][test_vect])**2
        #             A_suma_fin=A_suma_fin+A_suma
        #         A_euqlidean_distance = sqrt(A_suma_fin)
        #         if A_euqlidean_distance<A_min_dist:
        #             A_min_dist=A_euqlidean_distance
        #
        #     for Q_train_vect in range(0,len(QUERTUS_Training[0])):
        #         Q_euqlidean_distance=0
        #         Q_suma_fin=0
        #         for element in range (0,len(Combine_Test)):
        #             Q_suma=((QUERTUS_Training[element][Q_train_vect])-Combine_Test[element][test_vect])**2
        #             Q_suma_fin=Q_suma_fin+Q_suma
        #         Q_euqlidean_distance=sqrt(Q_suma_fin)
        #         if Q_euqlidean_distance < Q_min_dist:
        #             Q_min_dist = Q_euqlidean_distance
        #
        #     if (test_vect <= len(Acer_Test_number)) & (A_min_dist<Q_min_dist):
        #         NN_good_samples=NN_good_samples
        #     elif (test_vect >len(Acer_Test_number)) & (A_min_dist>Q_min_dist):
        #         NN_good_samples = NN_good_samples
        #     else:
        #         NN_good_samples=NN_good_samples-1
        #
        # efficiency=round((NN_good_samples/len(Combine_Test[0])*100),2)
        # print("NN_e", efficiency, "%")
        # return efficiency
#------------------------------------------------------k-NN----------------------------------------------------

    # if clasyficator == "k-NN":
    #     k_NN_good_samples=len(Combine_Test[0])
    #     for test_vect in range(0,len(Combine_Test[0])):
    #         K_NN_A_matrix = [1000] * k
    #         K_NN_Q_matrix = [1000] * k
    #         for A_train_vect in range(0,len(ACER_Training[0])):
    #             A_euqlidean_distance=0
    #             A_suma_fin=0
    #             for element in range(0,len(Combine_Test)):
    #                 A_suma=((ACER_Training[element][A_train_vect])-Combine_Test[element][test_vect])**2
    #                 A_suma_fin=A_suma_fin+A_suma
    #             A_euqlidean_distance = sqrt(A_suma_fin)
    #             # liczenie efektywnosci
    #             if A_euqlidean_distance<max(K_NN_A_matrix):
    #                 K_NN_A_matrix[K_NN_A_matrix.index(max(K_NN_A_matrix))]=A_euqlidean_distance
    #
    #             # K_NN_A_matrix.sort(reverse=False)
    #             # k_A_sum=sum((heapq.nsmallest(k, K_NN_A_matrix)))
    #
    #             # numpy.sort(K_NN_A_matrix)[:k])
    #             # k_A_sum = sum(numpy.sort(K_NN_A_matrix)[:k])
    #
    #         for Q_train_vect in range(0,len(QUERTUS_Training[0])):
    #             Q_euqlidean_distance=0
    #             Q_suma_fin=0
    #             for element in range (0,len(Combine_Test)):
    #                 Q_suma=((QUERTUS_Training[element][Q_train_vect])-Combine_Test[element][test_vect])**2
    #                 Q_suma_fin=Q_suma_fin+Q_suma
    #             Q_euqlidean_distance=sqrt(Q_suma_fin)
    #             # najefektywniej zastepowac, zadne cuda z algorytmami sortujacymi nie daja rady
    #             if Q_euqlidean_distance<max(K_NN_Q_matrix):
    #                 K_NN_Q_matrix[K_NN_Q_matrix.index(max(K_NN_Q_matrix))]= Q_euqlidean_distance
    #             # K_NN_Q_matrix.append(Q_euqlidean_distance)
    #
    #         # k_Q_sum = sum((msort(K_NN_Q_matrix))[:k])
    #         # k_Q_sum = sum(heapq.nsmallest(k, K_NN_Q_matrix))
    #         # K_NN_Q_matrix.sort(reverse=True)
    #         # k_Q_sum = sum(numpy.sort(K_NN_A_matrix)[:k])
    #         k_A_sum = sum(K_NN_A_matrix)
    #         k_Q_sum = sum(K_NN_Q_matrix)
    #
    #         if (test_vect <= len(Acer_Test_number)) & (k_A_sum<k_Q_sum):
    #             k_NN_good_samples=k_NN_good_samples
    #         elif (test_vect >len(Acer_Test_number)) & (k_A_sum>k_Q_sum):
    #             k_NN_good_samples = k_NN_good_samples
    #         else:
    #             k_NN_good_samples=k_NN_good_samples-1
    #
    #
    #
    #     efficiency=round((k_NN_good_samples/len(Combine_Test[0])*100),2)
    #     print("k_NN_e", efficiency, "%")
    #     return efficiency
#---------------------------------------------------NM---------------------------------------------------------------

    # if clasyficator=="NM":
    #     NM_good_samples = len(Combine_Test[0])
    #
    #     A_training_mean=[]
    #     Q_training_mean=[]
    #
    #     for row in ACER_Training:
    #         A_mean=numpy.mean(row)
    #         A_training_mean.append(A_mean)
    #
    #     for row in QUERTUS_Training:
    #         Q_mean=numpy.mean(row)
    #         Q_training_mean.append(Q_mean)
    #
    #     for test_vect in range(0,len(Combine_Test[0])):
    #         A_euqlidean_distance=0
    #         A_suma_fin=0
    #         Q_euqlidean_distance = 0
    #         Q_suma_fin = 0
    #         A_min_mean = 1000
    #         Q_min_mean = 1000
    #         for element in range(0,len(Combine_Test)):
    #             A_suma=((A_training_mean[element])-Combine_Test[element][test_vect])**2
    #             A_suma_fin=A_suma_fin+A_suma
    #         A_euqlidean_distance = sqrt(A_suma_fin)
    #
    #         for element in range(0, len(Combine_Test)):
    #             Q_suma = ((Q_training_mean[element]) - Combine_Test[element][test_vect]) ** 2
    #             Q_suma_fin = Q_suma_fin + Q_suma
    #         Q_euqlidean_distance = sqrt(Q_suma_fin)
    #
    #         if (test_vect <= len(Acer_Test_number)) & (A_euqlidean_distance < Q_euqlidean_distance):
    #             NM_good_samples = NM_good_samples
    #         elif (test_vect > len(Acer_Test_number)) & (A_euqlidean_distance > Q_euqlidean_distance):
    #             NM_good_samples = NM_good_samples
    #         else:
    #             NM_good_samples = NM_good_samples - 1
    #
    #     efficiency = round((NM_good_samples / len(Combine_Test[0]) * 100), 2)
    #     print("NM_e", efficiency, "%")
    #     return efficiency

# FLD_listOfcombination(2)

# ---------------------------------------------------NN-------------------------------------------------------------------
def NN(Combine_Test,ACER_Training,QUERTUS_Training,Acer_Test_number):

    NN_good_samples = len(Combine_Test[0])

    for test_vect in range(0, len(Combine_Test[0])):
        A_min_dist = 1000
        Q_min_dist = 1000
        for A_train_vect in range(0, len(ACER_Training[0])):
            A_euqlidean_distance = 0
            A_suma_fin = 0
            for element in range(0, len(Combine_Test)):
                A_suma = ((ACER_Training[element][A_train_vect]) - Combine_Test[element][test_vect]) ** 2
                A_suma_fin = A_suma_fin + A_suma
            A_euqlidean_distance = sqrt(A_suma_fin)
            if A_euqlidean_distance < A_min_dist:
                A_min_dist = A_euqlidean_distance

        for Q_train_vect in range(0, len(QUERTUS_Training[0])):
            Q_euqlidean_distance = 0
            Q_suma_fin = 0
            for element in range(0, len(Combine_Test)):
                Q_suma = ((QUERTUS_Training[element][Q_train_vect]) - Combine_Test[element][test_vect]) ** 2
                Q_suma_fin = Q_suma_fin + Q_suma
            Q_euqlidean_distance = sqrt(Q_suma_fin)
            if Q_euqlidean_distance < Q_min_dist:
                Q_min_dist = Q_euqlidean_distance

        if (test_vect <= len(Acer_Test_number)) & (A_min_dist < Q_min_dist):
            NN_good_samples = NN_good_samples
        elif (test_vect > len(Acer_Test_number)) & (A_min_dist > Q_min_dist):
            NN_good_samples = NN_good_samples
        else:
            NN_good_samples = NN_good_samples - 1

    efficiency = round((NN_good_samples / len(Combine_Test[0]) * 100), 2)
    print("NN_e", efficiency, "%")
    return efficiency

# --------------------------------------------------k-NN---------------------------------------------------------------
def k_NN(Combine_Test,ACER_Training,QUERTUS_Training,Acer_Test_number,k):

    k_NN_good_samples = len(Combine_Test[0])

    for test_vect in range(0, len(Combine_Test[0])):
        K_NN_A_matrix = [1000] * k
        K_NN_Q_matrix = [1000] * k
        for A_train_vect in range(0, len(ACER_Training[0])):
            A_euqlidean_distance = 0
            A_suma_fin = 0
            for element in range(0, len(Combine_Test)):
                A_suma = ((ACER_Training[element][A_train_vect]) - Combine_Test[element][test_vect]) ** 2
                A_suma_fin = A_suma_fin + A_suma
            A_euqlidean_distance = sqrt(A_suma_fin)
            # liczenie efektywnosci
            if A_euqlidean_distance < max(K_NN_A_matrix):
                K_NN_A_matrix[K_NN_A_matrix.index(max(K_NN_A_matrix))] = A_euqlidean_distance

                # K_NN_A_matrix.sort(reverse=False)
                # k_A_sum=sum((heapq.nsmallest(k, K_NN_A_matrix)))

                # numpy.sort(K_NN_A_matrix)[:k])
                # k_A_sum = sum(numpy.sort(K_NN_A_matrix)[:k])

        for Q_train_vect in range(0, len(QUERTUS_Training[0])):
            Q_euqlidean_distance = 0
            Q_suma_fin = 0
            for element in range(0, len(Combine_Test)):
                Q_suma = ((QUERTUS_Training[element][Q_train_vect]) - Combine_Test[element][test_vect]) ** 2
                Q_suma_fin = Q_suma_fin + Q_suma
            Q_euqlidean_distance = sqrt(Q_suma_fin)
            # najefektywniej zastepowac, zadne cuda z algorytmami sortujacymi nie daja rady
            if Q_euqlidean_distance < max(K_NN_Q_matrix):
                K_NN_Q_matrix[K_NN_Q_matrix.index(max(K_NN_Q_matrix))] = Q_euqlidean_distance
                # K_NN_Q_matrix.append(Q_euqlidean_distance)

        # k_Q_sum = sum((msort(K_NN_Q_matrix))[:k])
        # k_Q_sum = sum(heapq.nsmallest(k, K_NN_Q_matrix))
        # K_NN_Q_matrix.sort(reverse=True)
        # k_Q_sum = sum(numpy.sort(K_NN_A_matrix)[:k])


        # wersja 0 -->mapowanie

        # for l in range(0,k):
        #     K_NN_A_matrix[k].append("a")
        #     K_NN_Q_matrix[k].append("q")
        # print ("a",K_NN_A_matrix,"q",K_NN_Q_matrix)

        # wersja 1
        K_NN_A_matrix.sort(reverse=False)
        K_NN_Q_matrix.sort(reverse=False)

        probki = []
        for i in range(0, k):
            probki.append(Tag("A",K_NN_A_matrix[i]))
            probki.append(Tag("Q", K_NN_Q_matrix[i]))

        probki.sort(key=lambda x: x.value,reverse=False)

        k_A_sum=0
        k_Q_sum=0

        for i in range(0,k):
            if probki[i].tag == "A":
                k_A_sum+=1
            elif probki[i].tag == "Q":
                k_Q_sum+=1

        if (test_vect <= len(Acer_Test_number)) & (k_A_sum > k_Q_sum):
            k_NN_good_samples = k_NN_good_samples
        elif (test_vect > len(Acer_Test_number)) & (k_A_sum < k_Q_sum):
            k_NN_good_samples = k_NN_good_samples
        else:
            k_NN_good_samples = k_NN_good_samples - 1


    efficiency = round((k_NN_good_samples / len(Combine_Test[0]) * 100), 2)
    print("k_NN_e", efficiency, "%")
    return efficiency

class Tag:
    def __init__(self, tag, value):
        self.tag = tag
        self.value = value

# --------------------------------------------------NM-----------------------------------------------------------------
def NM (Combine_Test,ACER_Training,QUERTUS_Training,Acer_Test_number):

    NM_good_samples = len(Combine_Test[0])

    A_training_mean = []
    Q_training_mean = []

    for row in ACER_Training:
        A_mean = numpy.mean(row)
        A_training_mean.append(A_mean)

    for row in QUERTUS_Training:
        Q_mean = numpy.mean(row)
        Q_training_mean.append(Q_mean)

    for test_vect in range(0, len(Combine_Test[0])):
        A_euqlidean_distance = 0
        A_suma_fin = 0
        Q_euqlidean_distance = 0
        Q_suma_fin = 0
        A_min_mean = 1000
        Q_min_mean = 1000
        for element in range(0, len(Combine_Test)):
            try:
                A_suma = ((A_training_mean[element]) - Combine_Test[element][test_vect]) ** 2
                A_suma_fin = A_suma_fin + A_suma
            except:
                print("testvector, element", test_vect,element)
        A_euqlidean_distance = sqrt(A_suma_fin)

        for element in range(0, len(Combine_Test)):
            Q_suma = ((Q_training_mean[element]) - Combine_Test[element][test_vect]) ** 2
            Q_suma_fin = Q_suma_fin + Q_suma
        Q_euqlidean_distance = sqrt(Q_suma_fin)

        if (test_vect <= len(Acer_Test_number)) & (A_euqlidean_distance < Q_euqlidean_distance):
            NM_good_samples = NM_good_samples
        elif (test_vect > len(Acer_Test_number)) & (A_euqlidean_distance > Q_euqlidean_distance):
            NM_good_samples = NM_good_samples
        else:
            NM_good_samples = NM_good_samples - 1

    efficiency = round((NM_good_samples / len(Combine_Test[0]) * 100), 2)
    print("NM_e", efficiency, "%")
    return efficiency

# ----------------------------------------------k_NM-----------------------------------------------------------------
# nadpisuje combine i nie czysci

def k_NM(Combine_Test,Acer_Test_number):


    k_NM_good_samples = len(Combine_Test[0])
    A_claster=[]
    Q_claster=[]

    ct=Combine_Test[:]
    ct.append(list(range(0,len(ct[0]))))
    # randomowo wybieramy dwa punkty(dwa mody)
    A_random = numpy.random.choice(0,len(ct[0]))
    Q_random = numpy.random.choice(len(ct[0]))

    A_mean=[]
    Q_mean=[]



    for row in range(0,len(ct)):
        A_mean.append(ct[row][A_random])
        Q_mean.append((ct[row][Q_random]))

    print("am",A_mean)
    print("qm",Q_mean)
    koniec=false
    out_check_list=[]

    while (not(len(out_check_list)==len(A_mean)+len(Q_mean)-2)):

        print("1")
        out_check_list=[]
        A_groupe = []
        Q_groupe = []

        A_check_mean = numpy.array(A_mean)
        Q_check_mean = numpy.array(Q_mean)

        for test_vect in range(0, len(ct[0])):

            A_euqlidean_distance = 0
            A_suma_fin = 0
            Q_euqlidean_distance = 0
            Q_suma_fin = 0
            for element in range(0, len(ct)-1):
                try:
                    A_suma = ((A_mean[element]) - ct[element][test_vect]) ** 2
                    A_suma_fin = A_suma_fin + A_suma
                except:
                    print("testvector, element", test_vect,element)
            A_euqlidean_distance = sqrt(A_suma_fin)

            for element in range(0, len(ct)-1):
                Q_suma = ((Q_mean[element]) - ct[element][test_vect]) ** 2
                Q_suma_fin = Q_suma_fin + Q_suma
            Q_euqlidean_distance = sqrt(Q_suma_fin)

            if A_euqlidean_distance<Q_euqlidean_distance:
                A_groupe.append(ct[len(ct)-1][test_vect])
            else:
                Q_groupe.append(ct[len(ct)-1][test_vect])



        for i in range(0,len(ct)-1):
            A_mean[i]=numpy.mean(numpy.array(ct[i])[A_groupe])
            Q_mean[i]=numpy.mean(numpy.array(ct[i])[Q_groupe])

        # ogarnac jak wyjsc z petli.
        # przerwanie petli kiedy wartosci sredniej nie zmienia sie o zadany prog
        for r in range(0,len(A_mean)-1):
            if abs(A_mean[r]-A_check_mean[r])<0.00001:
                out_check_list.append(1)
        for r in range(0,len(Q_mean)-1):
            if abs(Q_mean[r]-Q_check_mean[r])<0.00001:
                out_check_list.append(1)
        # print("check",out_check_list)
    # print("asrednia",A_mean)
    # print("Qsrdnia",Q_mean)
    for sample in A_groupe:
        if sample >= len (Acer_Test_number):
            k_NM_good_samples=k_NM_good_samples-1
    for sample in Q_groupe:
        if sample< len(Acer_Test_number):
            k_NM_good_samples=k_NM_good_samples-1

    print(k_NM_good_samples)
    efficiency = round((k_NM_good_samples / len(ct[0]) * 100), 2)
    print("k_NM_e", efficiency, "%")
    return efficiency


    # print("grupa",A_groupe)
    # print("agrupw",Q_groupe)

    # print("a",A_random)
    # print("q",Q_random)


    # print ("ct",ct)

# k_NM()

# -----------------------------------------klasyfikatory-----------------------------------------------------------
def clasyficator_calculation(clasyficator,k):
    global Train_Test_dictionary
    efficiency=0
    Combine_Test=Train_Test_dictionary["Combine_Test"]
    ACER_Training=Train_Test_dictionary["ACER_Training"]
    QUERTUS_Training = Train_Test_dictionary["Quertus_Trainig"]
    Acer_Test_number=Train_Test_dictionary["ACER_Test"][0]

    if clasyficator=="NN":
        return NN(Combine_Test,ACER_Training,QUERTUS_Training,Acer_Test_number)# trzeba dodać rozroznienia pomiedzi acerem i Quertusem
    if clasyficator=="k-NN":
        return k_NN(Combine_Test,ACER_Training,QUERTUS_Training,Acer_Test_number,k)
    if clasyficator=="NM":
        return NM(Combine_Test, ACER_Training, QUERTUS_Training, Acer_Test_number)
    if clasyficator == "k-NM":
        return(k_NM(Combine_Test, Acer_Test_number))

# clasyficator_calculation("k-NN",3)

# ---------------------------------KROSWALIDACJA---------------------------------------------------------------------
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def Crosvalid_Test_Training(clasyficator,k,r):
    ACER_clas = ACER.tolist()
    QUERTUS_clas = QUERTUS.tolist()
    ACER_manipultaion_matrix = []
    QUERTUS_manipulation_matrix = []
    # r=5
    for n in SFS_index:
        ACER_manipultaion_matrix.append(ACER_clas[n])
        QUERTUS_manipulation_matrix.append(QUERTUS_clas[n])

    A_range_len_1=int(len(ACER_manipultaion_matrix[0])/r)
    Q_range_len_1=int(len(QUERTUS_manipulation_matrix[0])/r)
    print("przedzialy",A_range_len_1)
    Quality_table=[]

    # ------------------------podprzedziały w rzedach------------------------------------------
    # print("przedziały",A_range_len_1)
    # splited_A_manipulation_matrix=[]
    # for c in range(0,len (ACER_manipultaion_matrix)):
    #     splited_A_manipulation_matrix.append([ACER_manipultaion_matrix[c][i:i + A_range_len_1] for i in range(0, len(ACER_manipultaion_matrix[c]), A_range_len_1)])
    #
    #     # chunks(ACER_manipultaion_matrix[c],5)
    #
    # print (ACER_manipultaion_matrix)
    # print(splited_A_manipulation_matrix)
    # -------------------------------------------bez podpodzialow---------------------------------------------------


    Acer_training_matrix = [[] for i in range(len(ACER_manipultaion_matrix))]
    # Acer_test_matrix = []
    Quertus_training_matrix = [[] for i in range(len(QUERTUS_manipulation_matrix))]
    # Quertus_test_matrix = []
    A_range_len=A_range_len_1
    Q_range_len=Q_range_len_1
    a=0
    q=0
    while (A_range_len<=len(ACER_manipultaion_matrix[0])-1):
        Acer_test_matrix = []
        Quertus_test_matrix = []
        A_index_test = range(a, A_range_len)
        print("przedziałA",A_index_test)
        Q_index_test = range(q, Q_range_len)
        A_index_train=[s for j, s in enumerate(range(0,len(ACER_manipultaion_matrix[0])-1)) if j not in A_index_test]
        Q_index_train=[u for f, u in enumerate(range(0,len(QUERTUS_manipulation_matrix[0])-1)) if f not in Q_index_test]


        for x in range(0,len(ACER_manipultaion_matrix)):
            for index in A_index_train:
                z=ACER_manipultaion_matrix[x][index]
                Acer_training_matrix[x].append(z)
            Acer_test_matrix.append([m for i, m in enumerate(ACER_manipultaion_matrix[x]) if i in A_index_test])

        for y in range(0,len(QUERTUS_manipulation_matrix)):
            for index in Q_index_train:
                z=QUERTUS_manipulation_matrix[y][index]
                Quertus_training_matrix[y].append(z)
            Quertus_test_matrix.append([m for i, m in enumerate(QUERTUS_manipulation_matrix[y]) if i in Q_index_test])

        # print("atr",Acer_training_matrix,"dlugosc",len(Acer_training_matrix[0]))
        # print("ates",Acer_test_matrix,"dlugosc",len(Acer_test_matrix[0]))

        Combine_test_matrix = [[] for f in range(len(Acer_test_matrix))]

        for t in range(0, len(Acer_test_matrix)):
            Combine_test_matrix[t] = Acer_test_matrix[t] + Quertus_test_matrix[t]

        if clasyficator == "NN":
            Quality_table.append(NN(Combine_test_matrix, Acer_training_matrix, Quertus_training_matrix,Acer_test_matrix[0])) # trzeba dodać rozroznienia pomiedzi acerem i Quertusem
        if clasyficator == "k-NN":
            Quality_table.append(k_NN(Combine_test_matrix, Acer_training_matrix, Quertus_training_matrix,Acer_test_matrix[0], k))
        if clasyficator == "NM":
            Quality_table.append(NM(Combine_test_matrix, Acer_training_matrix, Quertus_training_matrix,Acer_test_matrix[0]))
        if clasyficator=="k-NM":
            Quality_table.append(k_NM(Combine_test_matrix,Acer_test_matrix[0]))

        if A_range_len+A_range_len_1<len(ACER_manipultaion_matrix[0]):
            print("dlugosci",len(ACER_manipultaion_matrix[0]))
            A_range_len +=A_range_len_1
            Q_range_len +=Q_range_len_1
            a+=A_range_len_1
            q+=Q_range_len_1
        else:
            A_range_len += len(ACER_manipultaion_matrix[0])-1
            Q_range_len += len(QUERTUS_manipulation_matrix[0])-1
            da=(A_range_len+A_range_len_1)-len(ACER_manipultaion_matrix[0])
            dq=(Q_range_len+Q_range_len_1)-len(QUERTUS_manipulation_matrix[0])
            a += da
            q += dq

    print("tablica wynikow",Quality_table)
    print("srednia",numpy.mean(Quality_table))
    q=numpy.mean(Quality_table)
    return round(q,2)

    #print("dlugosc",len(Acer_test_matrix))
    #print("Acer:",len(Acer_test_matrix[0]))

# Crosvalid_Test_Training("NM",3,5)


# ------------------------------------------BOOTSTRAP----------------------------------------------------------------
def Bootstrap(clasyficator,k,i):

    ACER_clas = ACER.tolist()
    QUERTUS_clas = QUERTUS.tolist()
    ACER_manipultaion_matrix = []
    QUERTUS_manipulation_matrix = []
    # r=5
    for n in SFS_index:
        ACER_manipultaion_matrix.append(ACER_clas[n])
        QUERTUS_manipulation_matrix.append(QUERTUS_clas[n])

    iteration=0
    QualityTableB=[]
    while(iteration<i):

        Acer_test_matrix = []
        Quertus_test_matrix = []

        A_test_index=sample(range(0, len(ACER_manipultaion_matrix[0])), 100)
        Q_test_index=sample(range(0, len(QUERTUS_manipulation_matrix[0])), 100)

        for x in range(0,len(ACER_manipultaion_matrix)):
            Acer_test_matrix.append([m for i, m in enumerate(ACER_manipultaion_matrix[x]) if i in A_test_index])

        for y in range(0,len(QUERTUS_manipulation_matrix)):
            Quertus_test_matrix.append([m for i, m in enumerate(QUERTUS_manipulation_matrix[y]) if i in Q_test_index])

        Combine_test_matrix = [[] for f in range(len(Acer_test_matrix))]

        for t in range(0, len(Acer_test_matrix)):
            Combine_test_matrix[t] = Acer_test_matrix[t] + Quertus_test_matrix[t]

        if clasyficator == "NN":
            QualityTableB.append(NN(Combine_test_matrix, ACER_manipultaion_matrix, QUERTUS_manipulation_matrix,Acer_test_matrix[0])) # trzeba dodać rozroznienia pomiedzi acerem i Quertusem
        if clasyficator == "k-NN":
            QualityTableB.append(k_NN(Combine_test_matrix, ACER_manipultaion_matrix, QUERTUS_manipulation_matrix,Acer_test_matrix[0], k))
        if clasyficator == "NM":
            QualityTableB.append(NM(Combine_test_matrix, ACER_manipultaion_matrix, QUERTUS_manipulation_matrix,Acer_test_matrix[0]))
        if clasyficator=="k-NM":
            QualityTableB.append(k_NM(Combine_test_matrix,Acer_test_matrix[0]))
        iteration+=1

    print("tablica wynikow", QualityTableB)
    print("srednia", numpy.mean(QualityTableB))
    qB = numpy.mean(QualityTableB)
    return round(qB,2)

# Bootstrap("NM",1,3)

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


#FLD_listOfcombination(62)
#print(loadData("data.txt"))
#FSD()

